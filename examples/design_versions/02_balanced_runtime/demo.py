"""Balanced runtime demo.

Shows the four borrowed-from-pi-mono ideas in one run:

1. transform: drop noisy old turns before the LLM-boundary view
2. convert_to_llm: filter out UI-only messages at the boundary
3. steer(): inject a mid-run nudge between turns
4. AgentRuntime + subscribe(): observe lifecycle events live

Run:

    python3 examples/design_versions/02_balanced_runtime/demo.py
"""

from __future__ import annotations

from agent import AgentRuntime
from core import (
    Agent,
    Event,
    Message,
    State,
    default_convert_to_llm,
    last_message,
    message_text,
    print_trace,
    sequence,
)


def proposer(agent: Agent, visible: list[Message], state: State) -> Message:
    task = last_message(visible, kind="task").content
    return Message(
        role="assistant",
        content=f"For '{task}': use Agent + Message + State + run.",
        sender=agent.name,
        target="all",
        kind="proposal",
    )


def critic(agent: Agent, visible: list[Message], state: State) -> Message:
    proposal = last_message(visible, sender="proposer")
    nudge = next(
        (m.content for m in visible if m.kind == "steer"),
        "",
    )
    note = f" steer={nudge}" if nudge else ""
    return Message(
        role="assistant",
        content=f"Length={len(message_text(proposal))}.{note}",
        sender=agent.name,
        target="all",
        kind="critique",
    )


def judge(agent: Agent, visible: list[Message], state: State) -> Message:
    return Message(
        role="assistant",
        content="Final: schedule via NextFn; trace via Events; tweak via steer().",
        sender=agent.name,
        target="user",
        kind="final",
    )


def keep_recent(messages: list[Message]) -> list[Message]:
    """Transform: drop blocked messages before convert_to_llm sees them."""
    return [m for m in messages if m.kind != "blocked"]


def main() -> None:
    agents = [
        Agent("proposer", "proposes a design", proposer),
        Agent("critic", "checks the proposal", critic),
        Agent("judge", "writes the final answer", judge),
    ]

    runtime = AgentRuntime(
        agents,
        transform=keep_recent,
        convert_to_llm=default_convert_to_llm,
    )

    seen: list[str] = []
    runtime.subscribe(lambda event: seen.append(event.kind))

    stream = runtime.prompt(
        "Design a balanced multi-agent runtime.",
        target="proposer",
        next_agent=sequence("proposer", "critic", "judge"),
    )

    for event in stream:
        if event.kind == "turn_end" and event.payload.get("agent") == "proposer":
            runtime.steer(
                Message(
                    role="user",
                    content="be terse",
                    sender="user",
                    target="critic",
                    kind="steer",
                )
            )

    print_trace(runtime.state)
    print(f"\nevent kinds observed: {seen}")
    print(f"last LLM payload size: {len(runtime.state.data.get('last_llm_payload', []))}")


if __name__ == "__main__":
    main()
