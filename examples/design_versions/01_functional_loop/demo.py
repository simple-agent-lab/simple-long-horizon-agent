"""Functional agent-loop sketch.

Run:

    python3 examples/design_versions/01_functional_loop/demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union


Role = Literal["system", "user", "assistant", "tool"]
MessageContent = Union[str, list[dict[str, Any]]]
ModelFn = Callable[[list["Message"]], "Message"]

ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class Message:
    role: Role
    content: MessageContent = ""
    sender: str = ""
    target: str = ""
    kind: str = "message"
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str) and not isinstance(self.content, list):
            raise TypeError("content must be str or list[dict]")
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be dict, got {type(self.data).__name__}")


def context_view(
    messages: list[Message],
    instruction: str,
    last: int | None = None,
) -> list[Message]:
    visible = messages[-last:] if last is not None else messages
    return [
        Message("system", instruction, sender="system", kind="instruction"),
        *visible,
    ]


def fake_model(messages: list[Message]) -> Message:
    assistant_turns = [
        message for message in messages if message.role == "assistant"
    ]
    if not assistant_turns:
        return Message(
            "assistant",
            "I inspected the task and will answer with the smallest useful loop.",
            sender="fake-model",
            kind="thought",
        )
    return Message(
        "assistant",
        "Final: context_view -> model -> record message -> stop.",
        sender="fake-model",
        kind="final",
    )


def as_agent_message(agent_name: str, output: Message) -> Message:
    return Message(
        output.role,
        output.content,
        sender=agent_name,
        target="user" if output.kind == "final" else agent_name,
        kind=output.kind,
        data={**output.data, "model_sender": output.sender},
    )


def run_loop(
    task: str,
    instruction: str,
    model: ModelFn,
    max_steps: int = 3,
) -> list[Message]:
    messages = [
        Message("user", task, sender="user", target="assistant", kind="task")
    ]
    for _ in range(max_steps):
        output = model(context_view(messages, instruction))
        message = as_agent_message("assistant", output)
        messages.append(message)
        if message.kind == "final":
            break
    return messages


def print_trace(messages: list[Message]) -> None:
    for index, message in enumerate(messages):
        print(f"{index:02d} {message.kind:<11} {message.sender:>10} {message.content}")


def main() -> None:
    messages = run_loop(
        "Design the simplest readable agent loop.",
        "You are a tiny teaching agent.",
        fake_model,
    )
    print_trace(messages)


if __name__ == "__main__":
    main()
