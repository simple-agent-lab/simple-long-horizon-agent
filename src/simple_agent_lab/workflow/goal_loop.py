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

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from simple_agent_lab.core import Agent
from simple_agent_lab.messages import AssistantMessage
from simple_agent_lab.protocols import GoalLifecycleStatus, GoalStatusEvent
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag

from .base import StepResult, as_text, final_output, never_abort

# Terminal subset of `GoalLifecycleStatus` (drops "active"): the statuses a
# finished goal loop can return.
GoalStatus = Literal["complete", "blocked", "budget_exhausted", "aborted"]

# Codex rule: same blocker ≥3 consecutive turns → report blocked
BLOCKED_STREAK_THRESHOLD = 3


@dataclass(frozen=True)
class GoalBudgets:
    """Multi-dimensional budget; each `None` means unbounded.

    `max_turns` bounds continuation turns. `token_budget` bounds cumulative
    output tokens (reads `AssistantMessage.usage.output_tokens`).
    `wall_clock_seconds` enforces a wall-time deadline (implemented as a
    composed abort, so it surfaces as `status="aborted"` not
    `status="budget_exhausted"`).
    """

    max_turns: int | None = None
    token_budget: int | None = None  # cumulative output tokens
    wall_clock_seconds: float | None = None


@dataclass(frozen=True)
class CompletionResult:
    """One verdict from a `CompletionCheck`.

    `done=True` stops the loop with `complete`. `blocked=True` (with a stable
    `reason`) feeds the >=3-consecutive-turn blocked-streak rule.
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
    tokens_used: int = 0


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


def _output_tokens(state: State) -> int:
    """Cumulative output tokens across all assistant messages on `state`.

    Tolerates `usage is None` turns (older/fake messages) without crashing.
    """
    total = 0
    for message in state.messages:
        if isinstance(message, AssistantMessage) and message.usage is not None:
            total += message.usage.output_tokens
    return total


def _record_goal_event(
    state: State,
    *,
    objective: str,
    status: GoalLifecycleStatus,
    turns_used: int,
    tokens_used: int = 0,
    reason: str = "",
) -> None:
    """Append an append-only `GoalStatusEvent` to `state.events`.

    This is the goal loop's event-sourced record: `state.record_event` stamps
    `index`/`elapsed` and appends. `StateSnapshot.apply` ignores it, so it lives
    in the trace log only (auditable, replay-able) and never enters the model
    context. Recorded each turn — `status="active"` while continuing, the
    terminal status on the final turn.
    """
    state.record_event(
        GoalStatusEvent(
            objective=objective,
            status=status,
            turns_used=turns_used,
            tokens_used=tokens_used,
            reason=reason,
        )
    )


def _drain(events, abort: AbortFlag) -> None:
    """Advance the inner loop to completion (the `run_agent` drain idiom)."""
    for _ in events:
        if abort():
            break


def _budget_hit(budgets: GoalBudgets, *, turns_used: int, tokens_used: int) -> bool:
    """Return True if the turns or token budget is exhausted."""
    if budgets.max_turns is not None and turns_used >= budgets.max_turns:
        return True
    if budgets.token_budget is not None and tokens_used >= budgets.token_budget:
        return True
    return False


def _wall_clock_abort(abort: AbortFlag, wall_clock_seconds: float | None) -> AbortFlag:
    """Compose the caller `abort` with a monotonic deadline (in_container pattern)."""
    if wall_clock_seconds is None:
        return abort
    deadline = time.monotonic() + wall_clock_seconds
    return lambda: abort() or time.monotonic() >= deadline


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
    prompt and re-check. Budgets (turns / cumulative output tokens / wall-clock)
    bound the loop. A `GoalStatusEvent` is appended to `state.events` every turn
    (event-sourced, replay-able).

    `inner_max_turns` bounds each individual `run`/`resume` of the inner ReAct
    loop (so a single continuation can't run forever); `budgets.max_turns`
    bounds how many continuation turns the goal loop itself takes.

    Wall-clock deadlines (via `budgets.wall_clock_seconds`) and explicit caller
    `abort` both surface as `status="aborted"`. Turns/token exhaustion surfaces
    as `status="budget_exhausted"`. The same blocker reported ≥3 consecutive
    turns surfaces as `status="blocked"`.
    """
    objective_text = as_text(objective)
    steps: list[StepResult] = []

    # Compose the caller's abort with the wall-clock deadline (if any).
    effective_abort = _wall_clock_abort(abort, budgets.wall_clock_seconds)

    state, events = agent.run(
        _goal_prompt(objective_text), max_turns=inner_max_turns, abort=effective_abort
    )
    _drain(events, effective_abort)
    turns_used = 0
    tokens_used = _output_tokens(state)
    steps.append(
        StepResult(
            name=agent.name,
            role=agent.role,
            task=objective_text,
            output=final_output(state, agent.name),
            state=state,
        )
    )
    _record_goal_event(
        state,
        objective=objective_text,
        status="active",
        turns_used=turns_used,
        tokens_used=tokens_used,
    )

    blocked_streak = 0
    last_blocker_reason: str = ""

    while True:
        # Check for caller abort or wall-clock deadline before evaluating check.
        if effective_abort():
            status: GoalStatus = "aborted"
            _record_goal_event(
                state,
                objective=objective_text,
                status=status,
                turns_used=turns_used,
                tokens_used=tokens_used,
            )
            return GoalResult(
                status=status,
                objective=objective_text,
                output=final_output(state, agent.name),
                steps=steps,
                turns_used=turns_used,
                tokens_used=tokens_used,
            )

        result = check(state)
        if result.done:
            status = "complete"
            _record_goal_event(
                state,
                objective=objective_text,
                status=status,
                turns_used=turns_used,
                tokens_used=tokens_used,
                reason=result.reason,
            )
            return GoalResult(
                status=status,
                objective=objective_text,
                output=final_output(state, agent.name),
                steps=steps,
                turns_used=turns_used,
                tokens_used=tokens_used,
            )

        # Track blocked streak: same reason ≥ BLOCKED_STREAK_THRESHOLD → blocked.
        if result.blocked:
            if result.reason == last_blocker_reason:
                blocked_streak += 1
            else:
                blocked_streak = 1
                last_blocker_reason = result.reason
            if blocked_streak >= BLOCKED_STREAK_THRESHOLD:
                status = "blocked"
                _record_goal_event(
                    state,
                    objective=objective_text,
                    status=status,
                    turns_used=turns_used,
                    tokens_used=tokens_used,
                    reason=last_blocker_reason,
                )
                return GoalResult(
                    status=status,
                    objective=objective_text,
                    output=final_output(state, agent.name),
                    steps=steps,
                    turns_used=turns_used,
                    tokens_used=tokens_used,
                )
        else:
            blocked_streak = 0
            last_blocker_reason = ""

        if _budget_hit(budgets, turns_used=turns_used, tokens_used=tokens_used):
            status = "budget_exhausted"
            _record_goal_event(
                state,
                objective=objective_text,
                status=status,
                turns_used=turns_used,
                tokens_used=tokens_used,
            )
            return GoalResult(
                status=status,
                objective=objective_text,
                output=final_output(state, agent.name),
                steps=steps,
                turns_used=turns_used,
                tokens_used=tokens_used,
            )

        state, events = agent.resume(
            state,
            _continuation_prompt(objective_text),
            max_turns=inner_max_turns,
            abort=effective_abort,
        )
        _drain(events, effective_abort)
        turns_used += 1
        tokens_used = _output_tokens(state)
        steps.append(
            StepResult(
                name=agent.name,
                role=agent.role,
                task=_continuation_prompt(objective_text),
                output=final_output(state, agent.name),
                state=state,
            )
        )
        _record_goal_event(
            state,
            objective=objective_text,
            status="active",
            turns_used=turns_used,
            tokens_used=tokens_used,
        )
