"""Tiny self-evolution probe for the balanced runtime.

Run from the repo root:

    PYTHONPATH=src python3 examples/design_versions/02_balanced_runtime/evolution_probe.py

This is not a full self-evolving agent. It is the smallest runnable harness
shape from ADR 0004:

    baseline run -> candidate run -> compare -> accept or reject

The evolved target here is a context transform. A later example can replace the
hard-coded candidate with a proposal agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core import (
    Agent,
    Message,
    State,
    TransformFn,
    assistant_message,
    last_message,
    print_trace,
    run_to_completion,
    sequence,
    system_message,
)


TASK = "Explain Python decorators in one clear sentence."


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    score: int
    draft: str
    state: State


def writer_step(agent: Agent, visible: list[Message], state: State) -> Message:
    wants_casual = any("casual" in str(m.content).lower() for m in visible)
    if wants_casual:
        content = "Decorators wrap functions: quick, tidy, reusable."
    else:
        content = (
            "Python decorators are a mechanism for modifying function behavior "
            "through wrapper functions."
        )
    return assistant_message(content, sender=agent.name, target="judge", kind="draft")


def judge_step(agent: Agent, visible: list[Message], state: State) -> Message:
    draft = last_message(visible, kind="draft")
    text = str(draft.content)
    score = int(len(text) <= 80 and "quick" in text.lower())
    reason = "short and concrete" if score else "too formal or too long"
    return assistant_message(
        f"score={score}: {reason}",
        sender=agent.name,
        target="user",
        kind="score",
        data={"score": score, "reason": reason},
    )


AGENTS = {
    "writer": Agent("writer", "Write one clear sentence.", writer_step),
    "judge": Agent("judge", "Score the draft.", judge_step),
}


def identity(messages: list[Message]) -> list[Message]:
    return messages


def inject_casual_lesson(messages: list[Message]) -> list[Message]:
    lesson = system_message(
        "Lesson: prefer a casual tone and concrete words.",
        sender="memory",
        target="writer",
        kind="lesson",
    )
    return [lesson, *messages]


def run_candidate(candidate_id: str, transform: TransformFn) -> CandidateResult:
    state = State(TASK, data={"candidate_id": candidate_id})
    state.send("task", "user", "writer", TASK)
    run_to_completion(
        AGENTS,
        state,
        sequence("writer", "judge"),
        transform=transform,
    )
    draft = last_message(state, kind="draft")
    score = last_message(state, kind="score")
    return CandidateResult(
        candidate_id=candidate_id,
        score=int(score.data["score"]),
        draft=str(draft.content),
        state=state,
    )


def main() -> None:
    candidates: list[tuple[str, Callable[[list[Message]], list[Message]]]] = [
        ("baseline", identity),
        ("casual_context", inject_casual_lesson),
    ]
    results = [
        run_candidate(candidate_id, transform)
        for candidate_id, transform in candidates
    ]
    baseline, challenger = results
    accepted = challenger if challenger.score > baseline.score else baseline

    print("candidate comparison")
    print("--------------------")
    for result in results:
        print(f"{result.candidate_id}: score={result.score} draft={result.draft}")
    print(f"\naccepted={accepted.candidate_id}")

    print("\naccepted trace")
    print_trace(accepted.state)


if __name__ == "__main__":
    main()
