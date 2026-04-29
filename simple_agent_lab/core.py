"""Tiny message runtime for multi-agent experiments.

The whole model is intentionally small:

    Agent + Message + State + context_view() + run()

Message is what agents exchange and what model adapters consume.
State.events is the trace of those messages. context_view() is the slice of
messages an agent can see for one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union


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


@dataclass(frozen=True)
class Event:
    step: int
    message: Message

    @property
    def kind(self) -> str:
        return self.message.kind

    @property
    def sender(self) -> str:
        return self.message.sender

    @property
    def target(self) -> str:
        return self.message.target

    @property
    def content(self) -> MessageContent:
        return self.message.content

    @property
    def data(self) -> dict[str, Any]:
        return self.message.data


@dataclass
class State:
    task: str
    events: list[Event] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[Message]:
        return [event.message for event in self.events]

    def send(
        self,
        kind: str,
        sender: str,
        target: str,
        content: MessageContent = "",
        role: str | None = None,
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
        event = Event(step=len(self.events), message=message)
        self.events.append(event)
        return message

    def by_kind(self, kind: str) -> list[Message]:
        return [message for message in self.messages if message.kind == kind]


ActFn = Callable[["Agent", list[Message], State], None]


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    act: ActFn


ViewFn = Callable[[Agent, State], list[Message]]


def context_view(agent: Agent, state: State, last: int | None = None) -> list[Message]:
    """Return the messages visible to this agent for its next step."""
    visible = [
        event.message
        for event in state.events
        if event.target in {agent.name, "all"} or event.sender == agent.name
    ]
    if last is not None:
        visible = visible[-last:]

    summary = state.data.get("summary")
    if summary and last is not None and len(visible) == last:
        return [
            Message(
                "system",
                str(summary),
                sender="state",
                target=agent.name,
                kind="summary",
            ),
            *visible,
        ]
    return visible


def run(
    agents: list[Agent],
    state: State,
    schedule: list[str],
    view: ViewFn = context_view,
) -> State:
    """Run agents in schedule order."""
    by_name = {agent.name: agent for agent in agents}
    for name in schedule:
        agent = by_name[name]
        agent.act(agent, view(agent, state), state)
    return state


def default_role(sender: str) -> str:
    if sender in {"system", "state"}:
        return "system"
    if sender == "user":
        return "user"
    return "assistant"


def model_message(message: Message, with_header: bool = True) -> ModelMessage:
    """Convert one lab message to a common chat-model message dict."""
    content = message.content
    has_lab_meta = bool(message.sender or message.target) or message.kind != "message"
    if with_header and has_lab_meta and isinstance(content, str):
        header = (
            f"[{message.sender} -> {message.target} | "
            f"{message.kind}/{message.channel}]"
        )
        content = f"{header}\n{content}"

    out: ModelMessage = {"role": message.role, "content": content}
    for key in ("name", "tool_call_id", "tool_calls"):
        if key in message.data:
            out[key] = message.data[key]
    return out


def model_messages(
    messages: list[Message],
    with_header: bool = True,
) -> list[ModelMessage]:
    return [model_message(message, with_header=with_header) for message in messages]


def message_text(message: Message) -> str:
    if isinstance(message.content, str) and message.content:
        return message.content.replace("\n", " ")[:120]
    if message.content:
        return str(message.content)[:120]
    return str(message.data)[:120]


def event_text(event: Event) -> str:
    return message_text(event.message)


def last_message(
    messages: State | list[Message],
    *,
    kind: str | None = None,
    sender: str | None = None,
) -> Message:
    source = messages.messages if isinstance(messages, State) else messages
    for message in reversed(source):
        if kind is not None and message.kind != kind:
            continue
        if sender is not None and message.sender != sender:
            continue
        return message
    raise LookupError(f"No message found for kind={kind!r}, sender={sender!r}")


def last_event(
    events: State | list[Event],
    *,
    kind: str | None = None,
    sender: str | None = None,
) -> Event:
    source = events.events if isinstance(events, State) else events
    for event in reversed(source):
        if kind is not None and event.kind != kind:
            continue
        if sender is not None and event.sender != sender:
            continue
        return event
    raise LookupError(f"No event found for kind={kind!r}, sender={sender!r}")


def print_trace(state: State) -> None:
    print("\ntrace")
    print("-----")
    for event in state.events:
        text = event_text(event)
        print(
            f"{event.step:02d} {event.kind:<10} "
            f"{event.sender:>12} -> {event.target:<12} {text}"
        )
