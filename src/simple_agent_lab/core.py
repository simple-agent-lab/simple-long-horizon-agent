"""Balanced message runtime for Simple Agent Lab.

The core model is still small:

    Agent + Message + State + context_view() + run()

Compared with the first tiny runtime, `run()` is now a generator that records
events, uses a `next_agent(state)` scheduler, exposes request/response
trace events, and can dispatch tool calls. Mid-run injection queues are
deliberately left out of this canonical version; callers add follow-up messages
explicitly to `State` and call `resume()` when they want another run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Union, cast

from .llm import (
    LLMRequest,
    Provider as LLMProvider,
    complete as llm_complete,
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .context_view import ContextPolicy, ContextView, build_context_view
from .messages import (
    AgentName,
    AssistantMessage,
    Message,
    MessageChannel,
    ContentInput,
    MessageKind,
    MessageRole,
    Role,
    ToolCallBlock,
    assistant_message,
    message_text,
    message_tool_calls,
    system_message,
    tool_result_message,
    user_message,
)
from .tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    text_result,
    tool_result_text,
)


EventKind = str


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
        return [
            event.message
            for event in self.events
            if event.message is not None
        ]

    def emit(self, kind: EventKind, **payload: Any) -> Event:
        event = Event(len(self.events), kind, dict(payload))
        self.events.append(event)
        return event

    def record(self, message: Message) -> Event:
        return self.emit("message", message=message)

    def send(
        self,
        kind: MessageKind,
        sender: AgentName,
        target: AgentName,
        content: ContentInput = "",
        role: MessageRole | None = None,
        channel: MessageChannel = "main",
        **data: Any,
    ) -> Message:
        message = make_message(
            role or default_role(sender),
            content,
            sender=sender,
            target=target,
            kind=kind,
            channel=channel,
            **data,
        )
        self.record(message)
        return message

    def by_kind(self, kind: str) -> list[Message]:
        return [message for message in self.messages if message.kind == kind]


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
    return list(build_agent_context_view(agent, state, last=last, policy=policy).messages)


def make_tool_result_message(
    call_id: str,
    tool_name: str,
    result: ToolResult,
    *,
    target: str,
) -> Message:
    """Convert a ToolResult into a transcript tool-result message."""
    return tool_result_message(
        tool_result_text(result),
        tool_call_id=call_id,
        tool_name=tool_name,
        target=target,
        is_error=result.is_error,
        data={
            "details": result.details,
            "content_blocks": [
                {"kind": c.kind, "text": c.text, "image_url": c.image_url}
                for c in result.content
            ],
        },
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

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
            "tool_execution_start",
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
                    "tool_execution_update",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    partial=partial,
                )
            yield state.emit(
                "tool_execution_end",
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                is_error=result.is_error,
                terminate=result.terminate,
            )

    for tool_call in tool_calls:
        yield state.record(
            make_tool_result_message(
                tool_call.id,
                tool_call.name,
                results[tool_call.id],
                target=target,
            )
        )


from collections.abc import Mapping

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
            if event.kind == "turn_end" and event.data.get("agent") == name
        )
        return name if turns < max_turns else None

    return next_agent


def _agent_dict(agents: dict[str, Agent] | list[Agent]) -> dict[str, Agent]:
    if isinstance(agents, dict):
        return agents
    return {agent.name: agent for agent in agents}


def _candidate_id(state: State) -> Any:
    return state.data.get("candidate_id")


def _message_outline(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "sender": message.sender,
            "target": message.target,
            "kind": message.kind,
            "channel": message.channel,
            "text": message_text(message),
        }
        for message in messages
    ]


def _tool_specs(tools: dict[str, AgentTool] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools.values()
    ]


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
    agent_by_name = _agent_dict(agents)
    tool_by_name = (
        tools
        if isinstance(tools, dict) or tools is None
        else {tool.name: tool for tool in tools}
    )
    if tool_by_name is not None:
        state.data["tools"] = tool_by_name

    yield state.emit("agent_start")
    while True:
        name = next_agent(state)
        if name is None:
            break
        if name not in agent_by_name:
            raise KeyError(f"Unknown agent {name!r}")

        agent = agent_by_name[name]
        yield state.emit("turn_start", agent=name)

        context = build_agent_context_view(
            agent,
            state,
            last=last,
            policy=context_policy,
        )
        visible = transform(list(context.messages))
        llm_payload = messages_to_llm_messages(visible, with_header=True)
        state.data["last_llm_payload"] = llm_payload

        candidate_id = _candidate_id(state)
        request_payload: dict[str, Any] = {
            "agent": name,
            "visible_count": len(visible),
            "llm_message_count": len(llm_payload),
            "visible": _message_outline(visible),
            "context_view": context.as_dict(),
            "tools": _tool_specs(tool_by_name),
            "llm_payload": llm_payload,
        }
        if candidate_id is not None:
            request_payload["candidate_id"] = candidate_id
        yield state.emit("model_request", **request_payload)

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
        yield state.emit("model_response", **response_payload)

        yield state.record(output)

        if tool_by_name and output_tool_calls:
            tool_terminated = False
            for event in dispatch_tool_calls(output, tool_by_name, state, abort=abort):
                yield event
                if event.kind == "tool_execution_end" and event.data.get("terminate"):
                    tool_terminated = True
            if tool_terminated:
                yield state.emit("turn_end", agent=name, terminated=True)
                yield state.emit("agent_end", reason="tool_terminate")
                return

        yield state.emit("turn_end", agent=name)

    yield state.emit("agent_end", reason="done")


def run_to_completion(
    agents: dict[str, Agent] | list[Agent],
    state: State,
    next_agent: NextFn,
    **kwargs: Any,
) -> State:
    """Drain run() and return the final State."""
    for _ in run(agents, state, next_agent, **kwargs):
        pass
    return state


Listener = Callable[[Event], None]


class AgentRuntime:
    """Small stateful wrapper around run().

    It owns State, a listener list, and a cancel flag. It intentionally does not
    own injection queues; extra user input should be recorded explicitly and then
    driven through `resume()`.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        transform: TransformFn = lambda messages: messages,
        last: int | None = None,
        context_policy: ContextPolicy | None = None,
        tools: list[AgentTool] | None = None,
    ) -> None:
        self._agents = {agent.name: agent for agent in agents}
        self._transform = transform
        self._last = last
        self._context_policy = context_policy
        self.tools: dict[str, AgentTool] = {tool.name: tool for tool in (tools or [])}
        self._listeners: list[Listener] = []
        self._aborted = False
        self.state = State(task="")

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def abort(self) -> None:
        self._aborted = True

    def prompt(
        self,
        task: str,
        *,
        target: str,
        next_agent: NextFn,
    ) -> Iterator[Event]:
        self.state = State(task=task)
        self.state.send("task", "user", target, task)
        return self._drive(next_agent)

    def resume(self, next_agent: NextFn) -> Iterator[Event]:
        if not self.state.messages:
            raise RuntimeError("Cannot resume: state has no messages")
        return self._drive(next_agent)

    def _drive(self, next_agent: NextFn) -> Iterator[Event]:
        self._aborted = False
        stream = run(
            self._agents,
            self.state,
            next_agent,
            transform=self._transform,
            last=self._last,
            context_policy=self._context_policy,
            tools=self.tools,
            abort=lambda: self._aborted,
        )
        for event in stream:
            for listener in list(self._listeners):
                listener(event)
            yield event
            if self._aborted:
                end_event = self.state.emit("agent_end", reason="aborted")
                for listener in list(self._listeners):
                    listener(end_event)
                yield end_event
                return


def default_role(sender: str) -> Role:
    if sender in {"system", "state", "runtime"}:
        return "system"
    if sender == "user":
        return "user"
    return "assistant"


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
    if role == "tool_result":
        tool_call_id = str(data.pop("tool_call_id", ""))
        tool_name = str(data.pop("tool_name", sender or ""))
        is_error = bool(data.pop("is_error", False))
        return tool_result_message(
            content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            sender=sender or tool_name,
            target=target,
            kind=kind,
            channel=channel,
            is_error=is_error,
            data=data,
        )
    raise ValueError(f"Unknown message role: {role!r}")


def event_text(event: Event) -> str:
    message = getattr(event, "message", None)
    if message is not None:
        return message_text(message)
    return " ".join(f"{key}={value}" for key, value in event.data.items())


def last_message(
    source: State | list[Message],
    *,
    kind: str | None = None,
    sender: str | None = None,
) -> Message:
    messages = source.messages if isinstance(source, State) else source
    for message in reversed(messages):
        if kind is not None and message.kind != kind:
            continue
        if sender is not None and message.sender != sender:
            continue
        return message
    raise LookupError(f"No message found for kind={kind!r}, sender={sender!r}")


def last_event(
    source: State | list[Event],
    *,
    kind: str | None = None,
    sender: str | None = None,
) -> Event:
    events = source.events if isinstance(source, State) else source
    for event in reversed(events):
        if kind is not None and event.kind != kind:
            continue
        if sender is not None:
            message = event.message
            event_sender = message.sender if message is not None else event.data.get("agent")
            if event_sender != sender:
                continue
        return event
    raise LookupError(f"No event found for kind={kind!r}, sender={sender!r}")


def print_trace(state: State) -> None:
    print("\ntrace")
    print("-----")
    for event in state.events:
        if event.kind == "message":
            message = event.message
            if message is None:
                continue
            route = f"{message.sender} -> {message.target}"
            print(
                f"{event.index:02d} {event.kind:<21} {message.kind:<10} "
                f"{route:<24} {message_text(message)}"
            )
            if isinstance(message, AssistantMessage):
                for thinking_block in message.thinking:
                    preview = thinking_block.text.replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    tag = "redacted_thinking" if thinking_block.redacted else "thinking"
                    print(f"   {tag:<21} {preview}")
        elif event.kind == "model_request":
            candidate = event.data.get("candidate_id")
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {event.kind:<21} "
                f"agent={event.data.get('agent')} "
                f"visible={event.data.get('visible_count')} "
                f"llm_messages={event.data.get('llm_message_count')}{suffix}"
            )
        elif event.kind == "model_response":
            candidate = event.data.get("candidate_id")
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {event.kind:<21} "
                f"agent={event.data.get('agent')} "
                f"kind={event.data.get('output_kind')} "
                f"target={event.data.get('target')} "
                f"tool_calls={event.data.get('tool_call_count')}{suffix}"
            )
        else:
            extras = " ".join(f"{key}={value}" for key, value in event.data.items())
            print(f"{event.index:02d} {event.kind:<21} {extras}")
