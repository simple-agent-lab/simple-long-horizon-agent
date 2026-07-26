"""Parallel-Distill-Refine (PDR) — sequential test-time scaling via distillation.

The sequential arm of test-time-compute scaling (Scaling Test-Time Compute for
Agentic Coding, arXiv:2604.16529), and the frontier counterpart of `run_chain`:
where a chain threads one stage's *raw* output into the next, PDR runs a *fan of*
`width` independent attempts each round, *distills* them into a compact findings
brief (promising hypotheses, progress made, dead ends), and conditions the next
round's attempts on that brief. After `rounds` rounds a finalizer writes the
answer from the accumulated brief.

Where pure parallel sampling spends compute on independent attempts and just
*selects* one, PDR spends compute *sequentially* and *reuses* distilled signal
across rounds — the distilled brief is the channel that carries progress from
one round's fan of attempts into the next.

Every attempt, distillation, and the finalizer is an ordinary `run_agent` step
recorded in `steps`, so per-run cost flows into the trace/metrics pipeline
unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.tools import AbortFlag

from .base import (
    StepResult,
    WorkflowResult,
    as_text,
    make_role_agent,
    never_abort,
    run_agent,
)
from .goal_loop import CompletionCheck
from .parallel import run_parallel

DISTILLER_ROLE = "Distill several attempts into a compact findings brief."
DISTILLER_SYSTEM_PROMPT = (
    "You distill several independent attempts at the same task into a compact "
    "findings brief that seeds the next round: capture the most promising "
    "hypotheses, what progress has been made, and which approaches failed or "
    "hit dead ends. Be concise and concrete — signal only, no filler."
)


def _conditioned_task(task: str, brief: str) -> str:
    """The round task: the original task plus the running findings brief."""
    if not brief.strip():
        return task
    return f"{task}\n\n<prior_findings>\n{brief}\n</prior_findings>"


def _distill_prompt(task: str, steps: list[StepResult], prior: str) -> str:
    attempts = "\n\n".join(
        f"--- Attempt {i} ---\n{step.output}" for i, step in enumerate(steps, start=1)
    )
    prior_block = f"\n\nFindings so far:\n{prior}" if prior.strip() else ""
    return (
        "Distill the independent attempts below into a compact findings brief "
        "(promising hypotheses, progress made, dead ends) to seed the next "
        f"round of attempts.\n\nTask:\n{task}{prior_block}\n\n{attempts}"
    )


def run_pdr(
    worker: Agent | Sequence[Agent],
    distiller: Agent,
    task: str,
    *,
    rounds: int = 3,
    width: int = 3,
    finalizer: Agent | None = None,
    check: CompletionCheck | None = None,
    worker_max_turns: int = 20,
    distiller_max_turns: int = 4,
    finalizer_max_turns: int = 20,
    max_concurrency: int = 8,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Run `rounds` of (parallel attempts -> distill), then finalize the answer.

    Each round runs `width` independent attempts (concurrently, via
    `run_parallel`) on the task conditioned on the running brief, then the
    `distiller` folds those attempts into an updated brief. After the last
    round, `finalizer` writes the final answer from the task plus the
    accumulated brief.

    `worker` accepts two forms: a single agent is replicated into `width`
    attempts (diversity rides on the provider's sampling temperature), or a
    sequence of distinct agents is the per-round attempt pool (then `width` is
    ignored and the sequence length is used). The sequence form is what lets a
    caller give each attempt an isolated workspace — e.g. SWE-bench binds each
    attempt agent to its own git worktree so concurrent edits never collide.
    The same attempt agents are reused across rounds, so an agent that must
    start each round from a clean slate should reset its workspace in its
    `init_state` hook (fired once per `agent.run`). `finalizer` defaults to the
    first attempt agent.

    `check` is an optional "done early" gate (the same `CompletionCheck` the goal
    loop uses, e.g. `command_verifier_check`). When given, each round's attempts
    are scanned right after they run; the FIRST one whose `check(state).done` is
    True is returned immediately, skipping the remaining rounds, the
    distillations, and the finalizer. This is the big saving for simple tasks: if
    round 1 already solves it, PDR stops instead of spending its full budget.

    `steps` are, in order, every round's attempts followed by that round's
    distillation, and finally the finalizer step. (On an early `check` hit,
    `steps` end at the winning round's attempts — no distill, no finalizer.)
    """
    if rounds < 1:
        raise ValueError("run_pdr requires rounds >= 1")
    if isinstance(worker, Agent):
        if width < 1:
            raise ValueError("run_pdr requires width >= 1")
        attempts: list[Agent] = [worker] * width
    else:
        attempts = list(worker)
        if not attempts:
            raise ValueError("run_pdr requires at least one worker")
    default_finalizer = attempts[0]

    task_text = as_text(task)
    steps: list[StepResult] = []
    brief = ""
    for _ in range(rounds):
        if abort():
            break
        round_result = run_parallel(
            attempts,
            _conditioned_task(task_text, brief),
            worker_max_turns=worker_max_turns,
            max_concurrency=max_concurrency,
            abort=abort,
        )
        steps.extend(round_result.steps)
        # Done-early gate: a verifiably-correct attempt ends the run now, skipping
        # the remaining rounds, the distillations, and the finalizer.
        if check is not None:
            for attempt in round_result.steps:
                if check(attempt.state).done:
                    return WorkflowResult(output=attempt.output, steps=steps)
        if abort():
            break
        distill_step = run_agent(
            distiller,
            _distill_prompt(task_text, round_result.steps, brief),
            max_turns=distiller_max_turns,
            abort=abort,
            role="distiller",
        )
        steps.append(distill_step)
        brief = distill_step.output

    final_step = run_agent(
        finalizer or default_finalizer,
        _conditioned_task(task_text, brief),
        max_turns=finalizer_max_turns,
        abort=abort,
        role="finalizer",
    )
    steps.append(final_step)
    return WorkflowResult(output=final_step.output, steps=steps)


def make_distiller_agent(
    provider: LLMProvider,
    *,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the round distiller `Agent` for `run_pdr`."""
    return make_role_agent(
        provider,
        name="distiller",
        role=DISTILLER_ROLE,
        system_prompt=DISTILLER_SYSTEM_PROMPT,
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
