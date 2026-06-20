"""Parallel-Distill-Refine (PDR) — sequential test-time scaling via distillation.

The sequential arm of test-time-compute scaling (Scaling Test-Time Compute for
Agentic Coding, arXiv:2604.16529), and the frontier counterpart of `run_chain`:
where a chain threads one stage's *raw* output into the next, PDR runs a *fan of*
`width` independent attempts each round, *distills* them into a compact findings
brief (promising hypotheses, progress made, dead ends), and conditions the next
round's attempts on that brief. After `rounds` rounds a finalizer writes the
answer from the accumulated brief.

Contrast with RTV: RTV spends compute *in parallel* and *selects* one rollout;
PDR spends compute *sequentially* and *reuses* distilled signal across rounds.
Run both against the same suite to compare the parallel and sequential axes of
test-time compute on the same quality-vs-cost frontier.

Every attempt, distillation, and the finalizer is an ordinary `run_agent` step
recorded in `steps`, so per-run cost flows into the trace/metrics pipeline
unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AbortFlag

from .base import StepResult, WorkflowResult, as_text, never_abort, run_agent
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
    worker: Agent,
    distiller: Agent,
    task: str,
    *,
    rounds: int = 3,
    width: int = 3,
    finalizer: Agent | None = None,
    worker_max_turns: int = 20,
    distiller_max_turns: int = 4,
    finalizer_max_turns: int = 20,
    max_concurrency: int = 8,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Run `rounds` of (parallel attempts -> distill), then finalize the answer.

    Each round runs `width` independent `worker` attempts (concurrently, via
    `run_parallel`) on the task conditioned on the running brief, then the
    `distiller` folds those attempts into an updated brief. After the last
    round, `finalizer` (defaulting to `worker`) writes the final answer from the
    task plus the accumulated brief.

    `steps` are, in order, every round's attempts followed by that round's
    distillation, and finally the finalizer step.
    """
    if rounds < 1:
        raise ValueError("run_pdr requires rounds >= 1")
    if width < 1:
        raise ValueError("run_pdr requires width >= 1")

    task_text = as_text(task)
    steps: list[StepResult] = []
    brief = ""
    for _ in range(rounds):
        if abort():
            break
        round_result = run_parallel(
            [worker] * width,
            _conditioned_task(task_text, brief),
            worker_max_turns=worker_max_turns,
            max_concurrency=max_concurrency,
            abort=abort,
        )
        steps.extend(round_result.steps)
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
        finalizer or worker,
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
    name: str = "distiller",
    role: str = DISTILLER_ROLE,
    system_prompt: str = DISTILLER_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the round distiller `Agent` for `run_pdr`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
