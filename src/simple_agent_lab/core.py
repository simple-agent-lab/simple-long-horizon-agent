"""Balanced message runtime for Simple Agent Lab.

The core model is small::

    Agent + Message + State + context_view() + run()

`run()` drives one agent as a generator: each turn, it builds a context view
of the visible messages, calls `agent.generate(...)`, records request/response
trace events, and dispatches any returned tool calls. The loop stops on the
first turn whose output is the agent's `final` message, or when `max_turns`
is exhausted (truncated runs surface as `agent_end(reason="max_turns")` so
traces can distinguish "agent decided to stop" from "ran out of budget").

`Agent.run(task)` is a convenience wrapper that seeds the state with a task
message and calls `run(self, state, ...)`, returning `(state, events)` so the
caller can stream events and still inspect the populated state.

Multi-agent flows are expressed as a parent agent that delegates through
`tools.task_tool([b, c, d])`: the parent picks one sub-agent via the
`subagent_type` enum and the chosen sub-agent's final message comes back as
the tool result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from .compression import maybe_compress_context
from .context_view import (
    ContextPolicy,
    build_context_view,
)
from .llm import messages_to_llm_messages
from .messages import (
    Message,
    ToolCallBlock,
    ToolResultBlock,
    message_text,
    message_tool_calls,
    tool_results_message,
)
from .protocols import (
    Event,
    ToolExecutionEndEvent,
)
from .state import State
from .tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result


# An Agent's `generate` produces the next message from the visible context.
# It takes no other arguments: name/role/tools/etc. are closed over at the
# point where the function is built (see `llm_agent.py` and the test fakes).
GenerateFn = Callable[[list[Message]], Message]


@dataclass
class Agent:
    name: str
    generate: GenerateFn
    role: str = ""
    # `tools` is a tuple to make explicit that tools are bound at construction:
    # `run()` snapshots them once per call, so mutating this attribute mid-run
    # has no effect on the in-flight loop.
    tools: tuple[AgentTool, ...] = ()
    context_policy: ContextPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            self.tools = tuple(self.tools)

    def run(
        self,
        task: str,
        *,
        max_turns: int = 10,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Drive this agent on `task` until it emits a final message.

        Returns `(state, events)`. Caller iterates `events` to advance the
        loop and inspects `state` for the message/event history. Callers
        that need to prepend extra messages (e.g. a sub-agent context
        prelude) can `state.record(...)` them after this call and before
        starting to iterate `events`.
        """
        state = State(task=task)
        state.send("task", "user", self.name, task)
        events = run(
            self,
            state,
            max_turns=max_turns,
            abort=abort,
        )
        return state, events


def run(
    agent: Agent,
    state: State,
    *,
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
) -> Iterator[Event]:
    """Run one agent as a generator until it emits `final` or hits `max_turns`.

    Each yielded `Event` is recorded in `state`. Multi-agent flows are
    expressed by giving `agent` a `task_tool` whose sub-agents each call
    their own `run()` inside the tool execute function.
    """
    name = agent.name
    tool_by_name = {tool.name: tool for tool in agent.tools}

    yield state.agent_start()
    final_emitted = False
    for _ in range(max_turns):
        yield state.turn_start(agent=name)

        policy = agent.context_policy or ContextPolicy()
        for compression_event in maybe_compress_context(agent, state, policy):
            yield compression_event

        context = build_context_view(
            name,
            state.active_context_messages(),
            policy=policy,
        )
        visible = list(context.messages)
        llm_payload = messages_to_llm_messages(visible, with_header=True)
        state.data["last_llm_payload"] = llm_payload

        # `state.data` is the open experiment bag (dict[str, Any]); pin
        # the runtime-side projection of `candidate_id` to its real type
        # so the typed event/state APIs are reached with a real `str | None`.
        raw_candidate = state.data.get("candidate_id")
        candidate_id: str | None = (
            str(raw_candidate) if raw_candidate is not None else None
        )
        yield state.model_request(
            agent=name,
            visible_count=len(visible),
            llm_message_count=len(llm_payload),
            visible=[
                {
                    "role": message.role,
                    "sender": message.sender,
                    "target": message.target,
                    "kind": message.kind,
                    "channel": message.channel,
                    "text": message_text(message),
                }
                for message in visible
            ],
            context_view=context.as_dict(),
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tool_by_name.values()
            ],
            llm_payload=llm_payload,
            candidate_id=candidate_id,
        )

        output = agent.generate(visible)
        output_tool_calls = message_tool_calls(output)
        yield state.model_response(
            agent=name,
            output_kind=output.kind,
            target=output.target,
            tool_call_count=len(output_tool_calls),
            candidate_id=candidate_id,
        )

        yield state.record(output)

        if output.sender == name and output.kind == "final":
            final_emitted = True

        if tool_by_name and output_tool_calls:
            tool_terminated = False
            for event in dispatch_tool_calls(output, tool_by_name, state, abort=abort):
                yield event
                if isinstance(event, ToolExecutionEndEvent) and event.terminate:
                    tool_terminated = True
            if tool_terminated:
                yield state.turn_end(agent=name, terminated=True)
                yield state.agent_end(reason="tool_terminate")
                return

        yield state.turn_end(agent=name)

        if final_emitted:
            break

    yield state.agent_end(reason="done" if final_emitted else "max_turns")


def dispatch_tool_calls(
    assistant_msg: Message,
    tools: dict[str, AgentTool],
    state: State,
    *,
    abort: AbortFlag = lambda: False,
    max_concurrency: int = 8,
) -> Iterator[Event]:
    """Run assistant tool calls and append deterministic tool-result messages."""
    tool_calls = message_tool_calls(assistant_msg)
    if not tool_calls:
        return

    target = assistant_msg.sender or "agent"
    sequential = any(
        (tool := tools.get(tool_call.name)) is not None
        and tool.execution_mode == "sequential"
        for tool_call in tool_calls
    )
    workers = 1 if sequential else min(max_concurrency, len(tool_calls))

    for tool_call in tool_calls:
        yield state.tool_execution_start(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
        )

    update_buffers: dict[str, list[ToolResult]] = {
        tool_call.id: [] for tool_call in tool_calls
    }

    def make_on_update(call_id: str) -> ToolUpdateFn:
        def on_update(partial: ToolResult) -> None:
            update_buffers[call_id].append(partial)

        return on_update

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, ToolResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_tool_call = {
            pool.submit(
                _execute_one,
                tool_call,
                tools,
                abort,
                make_on_update(tool_call.id),
            ): tool_call
            for tool_call in tool_calls
        }

        for future in as_completed(future_to_tool_call):
            tool_call = future_to_tool_call[future]
            try:
                result = future.result()
            except Exception as exc:
                result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)
            results[tool_call.id] = result

            for partial in update_buffers[tool_call.id]:
                yield state.tool_execution_update(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    partial=partial,
                )
            yield state.tool_execution_end(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                is_error=result.is_error,
                terminate=result.terminate,
            )

    bundle = tool_results_message(
        [
            ToolResultBlock(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=tuple(results[tool_call.id].content),
                is_error=results[tool_call.id].is_error,
            )
            for tool_call in tool_calls
        ],
        target=target,
        data={
            "details": {
                tool_call.id: results[tool_call.id].details for tool_call in tool_calls
            },
        },
    )
    yield state.record(bundle)


def _execute_one(
    tool_call: ToolCallBlock,
    tools: dict[str, AgentTool],
    abort: AbortFlag,
    on_update: ToolUpdateFn | None,
) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return text_result(f"Tool {tool_call.name!r} not found", is_error=True)
    execute = tool.execute
    if execute is None:
        return text_result(
            f"Tool {tool_call.name!r} has no execute function",
            is_error=True,
        )

    def run_tool() -> ToolResult:
        try:
            return execute(tool_call.id, dict(tool_call.arguments), abort, on_update)
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    if tool.timeout_seconds is None:
        return run_tool()

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(run_tool)
        try:
            return future.result(timeout=tool.timeout_seconds)
        except FuturesTimeoutError:
            return text_result(
                f"Tool {tool_call.name!r} timed out after {tool.timeout_seconds}s",
                is_error=True,
            )
    finally:
        pool.shutdown(wait=False)
