"""Balanced message runtime for Simple Agent Lab.

The core model is small:

    Agent + Message + State + context_view() + run()

`run()` is a generator that records events, uses a `next_agent(state)`
scheduler, exposes request/response trace events, and can dispatch tool
calls. `Agent.run()` is the single-agent shortcut that drives the loop
with `until_final` and returns `(state, events)` so callers can stream
events and still inspect the populated state. `tools.task_tool([b, c, d])`
bundles sub-agents as a single dispatch tool: the parent picks one via
the `subagent_type` enum and the chosen sub-agent's final message comes
back as the tool result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

from .compression import maybe_compress_context
from .context_view import (
    ContextPolicy,
    build_context_view,
)
from .llm import messages_to_llm_messages
from .messages import (
    AgentName,
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
    TurnEndEvent,
)
from .state import State
from .tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result


StepFn = Callable[["Agent", list[Message], State], Message]
TransformFn = Callable[[list[Message]], list[Message]]
NextFn = Callable[[State], AgentName | None]


@dataclass
class Agent:
    name: str
    step: StepFn
    role: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    context_policy: ContextPolicy | None = None

    def run(
        self,
        task: str,
        *,
        max_turns: int = 10,
        transform: TransformFn = lambda messages: messages,
        last: int | None = None,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Drive this agent on `task` until it emits a final message.

        Returns `(state, events)`. Caller iterates `events` to advance the
        loop and inspects `state` for the message/event history.
        """
        state = State(task=task)
        state.send("task", "user", self.name, task)
        events = run(
            {self.name: self},
            state,
            until_final(self.name, max_turns=max_turns),
            transform=transform,
            last=last,
            abort=abort,
        )
        return state, events


def until_final(name: AgentName, *, max_turns: int = 3) -> NextFn:
    """Scheduler that re-runs `name` until it emits a final message or hits a turn cap."""

    def next_agent(state: State) -> AgentName | None:
        if any(
            message.sender == name and message.kind == "final"
            for message in state.messages
        ):
            return None
        turns = sum(
            1
            for event in state.events
            if isinstance(event, TurnEndEvent) and event.agent == name
        )
        return name if turns < max_turns else None

    return next_agent


def run(
    agents: dict[str, Agent] | list[Agent],
    state: State,
    next_agent: NextFn,
    *,
    transform: TransformFn = lambda messages: messages,
    last: int | None = None,
    abort: AbortFlag = lambda: False,
) -> Iterator[Event]:
    """Run agents as a generator. Each yielded Event is recorded in state."""
    agent_by_name = (
        agents if isinstance(agents, dict) else {agent.name: agent for agent in agents}
    )

    yield state.agent_start()
    while True:
        name = next_agent(state)
        if name is None:
            break
        if name not in agent_by_name:
            raise KeyError(f"Unknown agent {name!r}")

        agent = agent_by_name[name]
        tool_by_name = {tool.name: tool for tool in agent.tools}
        if tool_by_name:
            state.data["tools"] = tool_by_name
        else:
            state.data.pop("tools", None)
        yield state.turn_start(agent=name)

        resolved_policy = _resolve_context_policy(agent.context_policy, last)
        compression_events = maybe_compress_context(
            agent,
            state,
            resolved_policy,
        )
        for compression_event in compression_events:
            yield compression_event

        context = build_context_view(
            agent.name,
            state.active_context_messages(),
            policy=resolved_policy,
        )
        visible = transform(list(context.messages))
        llm_payload = messages_to_llm_messages(visible, with_header=True)
        state.data["last_llm_payload"] = llm_payload

        candidate_id = state.data.get("candidate_id")
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
                for tool in (tool_by_name or {}).values()
            ],
            llm_payload=llm_payload,
            candidate_id=candidate_id,
        )

        output = agent.step(agent, visible, state)
        output_tool_calls = message_tool_calls(output)
        yield state.model_response(
            agent=name,
            output_kind=output.kind,
            target=output.target,
            tool_call_count=len(output_tool_calls),
            candidate_id=candidate_id,
        )

        yield state.record(output)

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

    yield state.agent_end(reason="done")


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


def _resolve_context_policy(
    policy: ContextPolicy | None,
    last: int | None,
) -> ContextPolicy:
    resolved = policy or ContextPolicy()
    if last is not None:
        resolved = replace(resolved, last=last)
    return resolved
