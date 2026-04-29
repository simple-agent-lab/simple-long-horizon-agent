"""Event-sourced agent-loop core.

This version makes the runtime observable: model requests, model responses,
message records, and stop decisions are all runtime events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


Role = Literal["system", "user", "assistant", "tool"]
MessageContent = Union[str, list[dict[str, Any]]]
ModelMessage = dict[str, Any]
MetaMode = Literal["none", "header"]
RuntimeEventKind = Literal["message", "model_request", "model_response", "stop"]
StopReason = Literal["final", "max_steps"]

ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class Message:
    role: Role
    content: MessageContent = ""
    sender: str = ""
    target: str = ""
    kind: str = "message"
    channel: str = "main"
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str) and not isinstance(self.content, list):
            raise TypeError("content must be str or list[dict]")
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be dict, got {type(self.data).__name__}")


@dataclass(frozen=True)
class RuntimeEvent:
    index: int
    kind: RuntimeEventKind
    data: dict[str, Any]


@dataclass(frozen=True)
class Agent:
    name: str
    instruction: str


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 4
    last_messages: int | None = None
    meta: MetaMode = "header"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.last_messages is not None and self.last_messages < 1:
            raise ValueError("last_messages must be >= 1 when set")
        if self.meta not in {"none", "header"}:
            raise ValueError(f"meta must be 'none' or 'header', got {self.meta!r}")


@dataclass
class RuntimeState:
    task: str
    messages: list[Message] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)

    def emit(self, kind: RuntimeEventKind, **data: Any) -> RuntimeEvent:
        event = RuntimeEvent(len(self.events), kind, data)
        self.events.append(event)
        return event

    def record(self, message: Message) -> Message:
        self.messages.append(message)
        self.emit("message", message=message)
        return message

    def user(self, target: str, content: str, *, kind: str = "task") -> Message:
        return self.record(
            Message("user", content, sender="user", target=target, kind=kind)
        )


@dataclass(frozen=True)
class RunResult:
    state: RuntimeState
    steps: int
    stop_reason: StopReason


class ModelClient:
    def generate(self, messages: list[Message], meta: MetaMode = "none") -> Message:
        raise NotImplementedError


def context_view(agent: Agent, state: RuntimeState, config: RunConfig) -> list[Message]:
    visible = [
        message
        for message in state.messages
        if message.target in {agent.name, "all"} or message.sender == agent.name
    ]
    if config.last_messages is not None:
        visible = visible[-config.last_messages :]
    return [
        Message("system", agent.instruction, sender="system", target=agent.name, kind="instruction"),
        *visible,
    ]


def model_messages(messages: list[Message], meta: MetaMode = "none") -> list[ModelMessage]:
    if meta not in {"none", "header"}:
        raise ValueError(f"meta must be 'none' or 'header', got {meta!r}")

    payload = []
    for message in messages:
        content = message.content
        if meta == "header" and isinstance(content, str):
            header = f"[{message.sender} -> {message.target} | {message.kind}/{message.channel}]"
            content = f"{header}\n{content}"
        item = {"role": message.role, "content": content}
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in message.data:
                item[key] = message.data[key]
        payload.append(item)
    return payload


def as_agent_message(agent: Agent, output: Message) -> Message:
    return Message(
        output.role,
        output.content,
        sender=agent.name,
        target="user" if output.kind == "final" else agent.name,
        kind=output.kind,
        data={**output.data, "model_sender": output.sender},
    )


class AgentLoop:
    def __init__(self, config: RunConfig = RunConfig()):
        self.config = config

    def run(self, agent: Agent, state: RuntimeState, model: ModelClient) -> RunResult:
        for step in range(1, self.config.max_steps + 1):
            visible = context_view(agent, state, self.config)
            state.emit("model_request", agent=agent.name, visible_count=len(visible))
            output = model.generate(visible, meta=self.config.meta)
            state.emit(
                "model_response",
                model_sender=output.sender,
                output_kind=output.kind,
            )
            message = as_agent_message(agent, output)
            state.record(message)
            if message.kind == "final":
                state.emit("stop", reason="final", steps=step)
                return RunResult(state, step, "final")
        state.emit("stop", reason="max_steps", steps=self.config.max_steps)
        return RunResult(state, self.config.max_steps, "max_steps")


def print_trace(state: RuntimeState) -> None:
    for event in state.events:
        if event.kind == "message":
            message = event.data["message"]
            print(
                f"{event.index:02d} message        "
                f"{message.kind:<8} {message.sender:>10} {message.content}"
            )
        else:
            print(f"{event.index:02d} {event.kind:<14} {event.data}")
