"""Balanced message runtime for Simple Agent Lab.

The core model is still small:

    Agent + Message + State + context_view() + run()

Compared with the first tiny runtime, `run()` is now a generator that records
events, uses a `next_agent(state)` scheduler, exposes request/response
trace events, and can dispatch tool calls. Stateful conveniences such as
`AgentRuntime.resume()` live in `runtime.py` so the main runtime path stays
easy to inspect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterator, Union, cast

from .context_view import ContextPolicy, ContextView, build_context_view
from .llm import (
    LLMRequest,
    Provider as LLMProvider,
    complete as llm_complete,
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .messages import (
    AgentName,
    ContentInput,
    Message,
    MessageChannel,
    MessageKind,
    Role,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    message_text,
    message_tool_calls,
    system_message,
    tool_results_message,
    user_message,
)
from .tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    text_result,
)


class EventKind(str, Enum):
    MESSAGE = "message"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Event:
    index: int
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> Message | None:
        message = self.data.get("message")
        return message if message is not None else None


@dataclass
class State:
    task: str
    events: list[Event] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[Message]:
        return [event.message for event in self.events if event.message is not None]

    def emit(self, kind: EventKind, **payload: Any) -> Event:
        event = Event(len(self.events), kind, dict(payload))
        self.events.append(event)
        return event

    def record(self, message: Message) -> Event:
        return self.emit(EventKind.MESSAGE, message=message)

    def send(
        self,
        kind: MessageKind,
        sender: AgentName,
        target: AgentName,
        content: ContentInput = "",
        role: Role | None = None,
        channel: MessageChannel = "main",
        **data: Any,
    ) -> Message:
        resolved_role = role
        if resolved_role is None:
            if sender in {"system", "state", "runtime"}:
                resolved_role = "system"
            elif sender == "user":
                resolved_role = "user"
            else:
                resolved_role = "assistant"
        message = make_message(
            resolved_role,
            content,
            sender=sender,
            target=target,
            kind=kind,
            channel=channel,
            **data,
        )
        self.record(message)
        return message


StepFn = Callable[["Agent", list[Message], State], Message]


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    step: StepFn


TransformFn = Callable[[list[Message]], list[Message]]
NextFn = Callable[[State], AgentName | None]


def build_agent_context_view(
    agent: Agent,
    state: State,
    *,
    last: int | None = None,
    policy: ContextPolicy | None = None,
) -> ContextView:
    """Return the detailed context projection for an agent step.

    The positional `last` keeps the legacy "show only the last N visible
    messages" knob; if both `last` and `policy.last` are set, `last` wins.
    """
    resolved = policy or ContextPolicy()
    if last is not None:
        resolved = replace(resolved, last=last)
    return build_context_view(agent.name, state.messages, policy=resolved)


def context_view(
    agent: Agent,
    state: State,
    last: int | None = None,
    *,
    policy: ContextPolicy | None = None,
) -> list[Message]:
    """Return messages visible to this agent for its next step."""
    return list(
        build_agent_context_view(agent, state, last=last, policy=policy).messages
    )


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
        yield state.emit(
            EventKind.TOOL_EXECUTION_START,
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
                yield state.emit(
                    EventKind.TOOL_EXECUTION_UPDATE,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    partial=partial,
                )
            yield state.emit(
                EventKind.TOOL_EXECUTION_END,
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


RequestExtraSpec = Union[
    Mapping[str, Any],
    Callable[[list[Message]], dict[str, Any]],
]


def make_llm_step(
    provider: LLMProvider,
    *,
    system_prompt: str = "",
    request_extra: RequestExtraSpec | None = None,
    result_kind: str | None = None,
    target: str = "all",
) -> StepFn:
    """Build an Agent.step function backed by the shared LLM layer."""

    def resolve_request_extra(visible: list[Message]) -> dict[str, Any]:
        if request_extra is None:
            return {}
        if isinstance(request_extra, Mapping):
            # ty narrows to Mapping[Unknown, Unknown] after isinstance, so
            # cast back to the declared element types of RequestExtraSpec.
            return cast("dict[str, Any]", dict(request_extra))
        return dict(request_extra(visible))

    def step(agent: Agent, visible: list[Message], state: State) -> Message:
        tools = state.data.get("tools") or {}
        request = LLMRequest(
            provider=provider,
            messages=messages_to_llm_messages(visible),
            tools=[tool_to_llm_tool(tool) for tool in tools.values()],
            system_prompt=system_prompt or agent.role or None,
            extra=resolve_request_extra(visible),
        )
        response = llm_complete(request)
        kind = result_kind or (
            "final" if response.stop_reason == "end_turn" else "thought"
        )
        return llm_response_to_assistant_message(
            response,
            sender=agent.name,
            target=target,
            kind=kind,
        )

    return step


def sequence(*names: AgentName) -> NextFn:
    """Return a scheduler that runs the given agents once, in order."""
    iterator = iter(names)

    def next_agent(_: State) -> AgentName | None:
        return next(iterator, None)

    return next_agent


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
            if event.kind is EventKind.TURN_END and event.data.get("agent") == name
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
    context_policy: ContextPolicy | None = None,
    tools: dict[str, AgentTool] | list[AgentTool] | None = None,
    abort: AbortFlag = lambda: False,
) -> Iterator[Event]:
    """Run agents as a generator. Each yielded Event is recorded in state."""
    agent_by_name = (
        agents if isinstance(agents, dict) else {agent.name: agent for agent in agents}
    )
    tool_by_name = (
        tools
        if isinstance(tools, dict) or tools is None
        else {tool.name: tool for tool in tools}
    )
    if tool_by_name is not None:
        state.data["tools"] = tool_by_name

    yield state.emit(EventKind.AGENT_START)
    while True:
        name = next_agent(state)
        if name is None:
            break
        if name not in agent_by_name:
            raise KeyError(f"Unknown agent {name!r}")

        agent = agent_by_name[name]
        yield state.emit(EventKind.TURN_START, agent=name)

        context = build_agent_context_view(
            agent,
            state,
            last=last,
            policy=context_policy,
        )
        visible = transform(list(context.messages))
        llm_payload = messages_to_llm_messages(visible, with_header=True)
        state.data["last_llm_payload"] = llm_payload

        candidate_id = state.data.get("candidate_id")
        request_payload: dict[str, Any] = {
            "agent": name,
            "visible_count": len(visible),
            "llm_message_count": len(llm_payload),
            "visible": [
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
            "context_view": context.as_dict(),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in (tool_by_name or {}).values()
            ],
            "llm_payload": llm_payload,
        }
        if candidate_id is not None:
            request_payload["candidate_id"] = candidate_id
        yield state.emit(EventKind.MODEL_REQUEST, **request_payload)

        output = agent.step(agent, visible, state)
        output_tool_calls = message_tool_calls(output)
        response_payload: dict[str, Any] = {
            "agent": name,
            "output_kind": output.kind,
            "target": output.target,
            "tool_call_count": len(output_tool_calls),
        }
        if candidate_id is not None:
            response_payload["candidate_id"] = candidate_id
        yield state.emit(EventKind.MODEL_RESPONSE, **response_payload)

        yield state.record(output)

        if tool_by_name and output_tool_calls:
            tool_terminated = False
            for event in dispatch_tool_calls(output, tool_by_name, state, abort=abort):
                yield event
                if event.kind is EventKind.TOOL_EXECUTION_END and event.data.get(
                    "terminate"
                ):
                    tool_terminated = True
            if tool_terminated:
                yield state.emit(EventKind.TURN_END, agent=name, terminated=True)
                yield state.emit(EventKind.AGENT_END, reason="tool_terminate")
                return

        yield state.emit(EventKind.TURN_END, agent=name)

    yield state.emit(EventKind.AGENT_END, reason="done")


def make_message(
    role: str,
    content: ContentInput = "",
    *,
    sender: str = "",
    target: str = "",
    kind: str = "message",
    channel: str = "main",
    **data: Any,
) -> Message:
    """Construct the right role-specific Message variant."""
    if role == "user":
        return user_message(
            content,
            sender=sender or "user",
            target=target or "all",
            kind=kind,
            channel=channel,
            data=data,
        )
    if role == "system":
        return system_message(
            content,
            sender=sender or "system",
            target=target or "all",
            kind=kind,
            channel=channel,
            data=data,
        )
    if role == "assistant":
        return assistant_message(
            content,
            sender=sender or "assistant",
            target=target or "all",
            kind=kind,
            channel=channel,
            data=data,
        )
    raise ValueError(f"Unknown message role: {role!r}")
