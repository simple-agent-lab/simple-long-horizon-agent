"""Balanced multi-agent runtime core.

Borrowed from pi-mono:

- two-stage transform pipeline at the LLM boundary
  (transform: AgentMessage -> AgentMessage; convert_to_llm: AgentMessage -> dict)
- runtime is a generator that yields Events
- steering / follow-up dual queues
- before_act / after_act hooks with block/terminate semantics
- next_agent(state) -> str | None replaces the static schedule list
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Literal, Optional, Union


MessageContent = Union[str, list[dict[str, Any]]]
ModelMessage = dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: str
    content: MessageContent = ""
    sender: str = ""
    target: str = ""
    kind: str = "message"
    channel: str = "main"
    data: dict[str, Any] = field(default_factory=dict)


EventKind = Literal[
    "agent_start", "agent_end", "turn_start", "turn_end", "message", "blocked"
]


@dataclass(frozen=True)
class Event:
    step: int
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> Optional[Message]:
        return self.payload.get("message")


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
            if event.kind == "message" and event.message is not None
        ]

    def emit(self, kind: EventKind, **payload: Any) -> Event:
        event = Event(step=len(self.events), kind=kind, payload=payload)
        self.events.append(event)
        return event

    def record(self, message: Message) -> Event:
        return self.emit("message", message=message)

    def send(
        self,
        kind: str,
        sender: str,
        target: str,
        content: MessageContent = "",
        role: Optional[str] = None,
        channel: str = "main",
        **data: Any,
    ) -> Message:
        message = Message(
            role=role or default_role(sender),
            content=content,
            sender=sender,
            target=target,
            kind=kind,
            channel=channel,
            data=data,
        )
        self.record(message)
        return message


ActFn = Callable[["Agent", list[Message], State], Message]


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    act: ActFn


# ---------------------------------------------------------------------------
# Two-stage pipeline at the LLM boundary

TransformFn = Callable[[list[Message]], list[Message]]
ConvertToLlm = Callable[[list[Message]], list[ModelMessage]]


def context_view(agent: Agent, state: State, last: Optional[int] = None) -> list[Message]:
    """Return messages visible to this agent for its next step."""
    visible = [
        message
        for message in state.messages
        if message.target in {agent.name, "all"} or message.sender == agent.name
    ]
    if last is not None:
        visible = visible[-last:]
    return visible


def default_convert_to_llm(messages: list[Message]) -> list[ModelMessage]:
    """Drop UI-only kinds; map remaining messages to chat-model dicts."""
    out: list[ModelMessage] = []
    for message in messages:
        if message.kind in {"notification", "trace"}:
            continue
        out.append(model_message(message))
    return out


def model_message(message: Message, with_header: bool = True) -> ModelMessage:
    content = message.content
    has_meta = bool(message.sender or message.target) or message.kind != "message"
    if with_header and has_meta and isinstance(content, str):
        header = (
            f"[{message.sender} -> {message.target} | "
            f"{message.kind}/{message.channel}]"
        )
        content = f"{header}\n{content}"
    item: ModelMessage = {"role": message.role, "content": content}
    for key in ("name", "tool_call_id", "tool_calls"):
        if key in message.data:
            item[key] = message.data[key]
    return item


# ---------------------------------------------------------------------------
# Hooks


@dataclass(frozen=True)
class BeforeActResult:
    block: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class AfterActResult:
    content: Optional[MessageContent] = None
    terminate: bool = False


BeforeAct = Callable[[Agent, list[Message], State], Optional[BeforeActResult]]
AfterAct = Callable[[Agent, Message, State], Optional[AfterActResult]]


# ---------------------------------------------------------------------------
# Steering / follow-up queues

QueueMode = Literal["all", "one-at-a-time"]


@dataclass
class Queue:
    mode: QueueMode = "one-at-a-time"
    pending: list[Message] = field(default_factory=list)

    def push(self, message: Message) -> None:
        self.pending.append(message)

    def has_items(self) -> bool:
        return bool(self.pending)

    def drain(self) -> list[Message]:
        if not self.pending:
            return []
        if self.mode == "all":
            out, self.pending = self.pending, []
            return out
        head = [self.pending[0]]
        self.pending = self.pending[1:]
        return head

    def clear(self) -> None:
        self.pending = []


# ---------------------------------------------------------------------------
# Scheduling

NextFn = Callable[[State], Optional[str]]


def sequence(*names: str) -> NextFn:
    """Run the given agents once, in order."""
    iterator = iter(names)

    def _next(_: State) -> Optional[str]:
        return next(iterator, None)

    return _next


# ---------------------------------------------------------------------------
# Generator-based run


def run(
    agents: dict[str, Agent],
    state: State,
    next_agent: NextFn,
    *,
    transform: TransformFn = lambda messages: messages,
    convert_to_llm: Optional[ConvertToLlm] = None,
    last: Optional[int] = None,
    before_act: Optional[BeforeAct] = None,
    after_act: Optional[AfterAct] = None,
    steering: Optional[Queue] = None,
    follow_up: Optional[Queue] = None,
) -> Iterator[Event]:
    """Run the loop as a generator. Each yielded Event is also recorded in state."""
    yield state.emit("agent_start")
    while True:
        while True:
            if steering and steering.has_items():
                for message in steering.drain():
                    yield state.record(message)
            name = next_agent(state)
            if name is None:
                break
            agent = agents[name]
            yield state.emit("turn_start", agent=name)

            visible = context_view(agent, state, last=last)
            visible = transform(visible)
            if convert_to_llm is not None:
                state.data["last_llm_payload"] = convert_to_llm(visible)

            if before_act is not None:
                pre = before_act(agent, visible, state)
                if pre is not None and pre.block:
                    blocked = Message(
                        role="system",
                        content=pre.reason or "blocked",
                        sender="runtime",
                        target=name,
                        kind="blocked",
                    )
                    yield state.record(blocked)
                    yield state.emit("turn_end", agent=name, blocked=True)
                    continue

            reply = agent.act(agent, visible, state)

            if after_act is not None:
                post = after_act(agent, reply, state)
                if post is not None and post.content is not None:
                    reply = replace(reply, content=post.content)
                if post is not None and post.terminate:
                    yield state.record(reply)
                    yield state.emit("turn_end", agent=name, terminated=True)
                    yield state.emit("agent_end", reason="terminate")
                    return

            yield state.record(reply)
            yield state.emit("turn_end", agent=name)

        if follow_up and follow_up.has_items():
            for message in follow_up.drain():
                yield state.record(message)
            continue
        break

    yield state.emit("agent_end", reason="done")


def run_to_completion(
    agents: dict[str, Agent],
    state: State,
    next_agent: NextFn,
    **kwargs: Any,
) -> State:
    """Drain the run() generator and return the final State."""
    for _ in run(agents, state, next_agent, **kwargs):
        pass
    return state


# ---------------------------------------------------------------------------
# Helpers


def default_role(sender: str) -> str:
    if sender in {"system", "state", "runtime"}:
        return "system"
    if sender == "user":
        return "user"
    return "assistant"


def message_text(message: Message) -> str:
    if isinstance(message.content, str) and message.content:
        return message.content.replace("\n", " ")[:120]
    if message.content:
        return str(message.content)[:120]
    return str(message.data)[:120]


def last_message(
    source: Union[State, list[Message]],
    *,
    kind: Optional[str] = None,
    sender: Optional[str] = None,
) -> Message:
    messages = source.messages if isinstance(source, State) else source
    for message in reversed(messages):
        if kind is not None and message.kind != kind:
            continue
        if sender is not None and message.sender != sender:
            continue
        return message
    raise LookupError(f"No message found for kind={kind!r}, sender={sender!r}")


def print_trace(state: State) -> None:
    print("\ntrace")
    print("-----")
    for event in state.events:
        if event.kind == "message" and event.message is not None:
            message = event.message
            text = message_text(message)
            route = f"{message.sender} -> {message.target}"
            print(
                f"{event.step:02d} {event.kind:<11} {message.kind:<10} "
                f"{route:<24} {text}"
            )
        else:
            extras = " ".join(f"{k}={v}" for k, v in event.payload.items())
            print(f"{event.step:02d} {event.kind:<11} {extras}")
