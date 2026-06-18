"""Goal loop: autonomously continue a conversation until verifiably done.

`run_goal_loop` is the `/goal` "loop-engineering" primitive: it drives the
existing ReAct loop (`core.run`, via `Agent.run` / `Agent.resume`) and keeps
re-prompting the SAME conversation until an INDEPENDENT completion `check`
passes or a budget is exhausted. It owns only the outer structure + stop
condition (like every `workflow/` orchestrator); the per-agent turn/tool
handling is unchanged. Completion authority is pluggable (`CompletionCheck`)
rather than the LLM grading itself (the `run_reflection` anti-pattern).

Goal state is event-sourced: each turn appends a `GoalStatusEvent` to
`state.events` (`State.record_event`), so the goal lifecycle is auditable and
replay-able and rides along with `Agent.resume()` across sessions. It is NOT
stored in the mutable `state.data` scratchpad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from simple_agent_lab.core import Agent
from simple_agent_lab.protocols import GoalLifecycleStatus, GoalStatusEvent
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag

from .base import StepResult, as_text, final_output, never_abort

# Terminal subset of `GoalLifecycleStatus` (drops "active"): the statuses a
# finished goal loop can return.
GoalStatus = Literal["complete", "blocked", "budget_exhausted", "aborted"]


@dataclass(frozen=True)
class GoalBudgets:
    """Multi-dimensional budget; each `None` means unbounded.

    Phase 1 uses only `max_turns` (continuation turns). `token_budget`
    (cumulative output tokens) and `wall_clock_seconds` land in Phase 2.
    """

    max_turns: int | None = None


@dataclass(frozen=True)
class CompletionResult:
    """One verdict from a `CompletionCheck`.

    `done=True` stops the loop with `complete`. `blocked=True` (with a stable
    `reason`) feeds the >=3-consecutive-turn blocked-streak rule (Phase 2).
    """

    done: bool
    blocked: bool = False
    reason: str = ""


# A completion check inspects the (finished-this-turn) State and reports a
# verdict. It is INDEPENDENT of the agent's own claim of done-ness.
CompletionCheck = Callable[[State], CompletionResult]


@dataclass(frozen=True)
class GoalResult:
    """Outcome of a goal loop: terminal status + audit trail + counters."""

    status: GoalStatus
    objective: str
    output: str
    steps: list[StepResult] = field(default_factory=list)
    turns_used: int = 0


def _goal_prompt(objective: str) -> str:
    """First-turn prompt. Phase 3 wraps `objective` as untrusted data and adds
    the acknowledgment line; Phase 1 keeps it minimal."""
    return f"Work toward this objective until it is fully and verifiably done.\n\n{objective}"


def _continuation_prompt(objective: str) -> str:
    """Placeholder continuation nudge. Phase 3 replaces this with the
    Codex-grade completion-audit text."""
    return (
        "This is not done yet. Keep working toward the objective and verify "
        f"your work against the current state.\n\nObjective:\n{objective}"
    )


def _record_goal_event(
    state: State,
    *,
    objective: str,
    status: GoalLifecycleStatus,
    turns_used: int,
    reason: str = "",
) -> None:
    """Append an append-only `GoalStatusEvent` to `state.events`.

    This is the goal loop's event-sourced record: `state.record_event` stamps
    `index`/`elapsed` and appends. `StateSnapshot.apply` ignores it, so it lives
    in the trace log only (auditable, replay-able) and never enters the model
    context. Recorded each turn — `status="active"` while continuing, the
    terminal status on the final turn. (Phase 2 adds `tokens_used`.)
    """
    state.record_event(
        GoalStatusEvent(
            objective=objective,
            status=status,
            turns_used=turns_used,
            reason=reason,
        )
    )


def _drain(events, abort: AbortFlag) -> None:
    """Advance the inner loop to completion (the `run_agent` drain idiom)."""
    for _ in events:
        if abort():
            break


def _budget_hit(budgets: GoalBudgets, turns_used: int) -> bool:
    """Phase 1: only the continuation-turn budget. Extended in Phase 2."""
    return budgets.max_turns is not None and turns_used >= budgets.max_turns


def run_goal_loop(
    agent: Agent,
    objective: str,
    *,
    check: CompletionCheck,
    budgets: GoalBudgets = GoalBudgets(),
    abort: AbortFlag = never_abort,
    inner_max_turns: int = 20,
) -> GoalResult:
    """Drive `agent` toward `objective`, continuing until `check` reports done.

    Runs the agent once, then loops: evaluate `check(state)`; on `done` return
    `complete`; otherwise `resume` the SAME conversation with a continuation
    prompt and re-check. The continuation-turn budget bounds the loop. A
    `GoalStatusEvent` is appended to `state.events` every turn (event-sourced,
    replay-able).

    `inner_max_turns` bounds each individual `run`/`resume` of the inner ReAct
    loop (so a single continuation can't run forever); `budgets.max_turns`
    bounds how many continuation turns the goal loop itself takes.
    """
    objective_text = as_text(objective)
    steps: list[StepResult] = []

    state, events = agent.run(_goal_prompt(objective_text), max_turns=inner_max_turns, abort=abort)
    _drain(events, abort)
    turns_used = 0
    steps.append(
        StepResult(
            name=agent.name,
            role=agent.role,
            task=objective_text,
            output=final_output(state, agent.name),
            state=state,
        )
    )
    _record_goal_event(state, objective=objective_text, status="active",
                       turns_used=turns_used)

    while True:
        result = check(state)
        if result.done:
            status: GoalStatus = "complete"
            _record_goal_event(state, objective=objective_text, status=status,
                               turns_used=turns_used, reason=result.reason)
            return GoalResult(status=status, objective=objective_text,
                              output=final_output(state, agent.name), steps=steps,
                              turns_used=turns_used)

        if _budget_hit(budgets, turns_used):
            status = "budget_exhausted"
            _record_goal_event(state, objective=objective_text, status=status,
                               turns_used=turns_used)
            return GoalResult(status=status, objective=objective_text,
                              output=final_output(state, agent.name), steps=steps,
                              turns_used=turns_used)

        state, events = agent.resume(state, _continuation_prompt(objective_text),
                                     max_turns=inner_max_turns, abort=abort)
        _drain(events, abort)
        turns_used += 1
        steps.append(
            StepResult(
                name=agent.name,
                role=agent.role,
                task=_continuation_prompt(objective_text),
                output=final_output(state, agent.name),
                state=state,
            )
        )
        _record_goal_event(state, objective=objective_text, status="active",
                           turns_used=turns_used)
