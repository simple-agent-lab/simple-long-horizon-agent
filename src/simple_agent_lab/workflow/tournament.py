"""Recursive Tournament Voting (RTV) — parallel test-time scaling with selection.

A frontier alternative to `run_parallel`'s aggregator: instead of *synthesizing*
several independent rollouts into one answer, RTV *selects* the single best one
through a tournament of pairwise/small-group comparisons. This is the parallel
arm of test-time-compute scaling for agents (Scaling Test-Time Compute for
Agentic Coding, arXiv:2604.16529): run N independent rollouts, optionally
compress each into a structured summary, then recursively narrow the population
by comparing them in small groups until one survives.

Why selection instead of synthesis: an aggregator can blend a correct answer
with wrong ones and regress; a selector that picks one rollout never produces a
worse-than-its-best output. The cost is N rollouts plus O(N) cheap comparisons.

Every rollout and every comparison is an ordinary `run_agent` step recorded in
the result's `steps`, so the full per-rollout token/turn cost flows into the
trace/metrics pipeline unchanged — which is what lets RTV sit on a
quality-vs-cost frontier next to `single` / `parallel`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AbortFlag

from .base import (
    StepResult,
    WorkflowResult,
    as_text,
    never_abort,
    pick_index,
    run_agent,
)
from .parallel import run_parallel

SELECTOR_ROLE = "Pick the single best answer from several candidates."
SELECTOR_SYSTEM_PROMPT = (
    "You are a selection judge. You are given a task and several candidate "
    "answers produced independently. Decide which single candidate is best — "
    "most correct, complete, and well-supported. Do not blend them. Reply with "
    "ONLY the candidate number."
)

SUMMARIZER_ROLE = "Compress one rollout into a faithful comparison synopsis."
SUMMARIZER_SYSTEM_PROMPT = (
    "You compress a candidate answer into a compact, faithful synopsis for "
    "comparison against other candidates: preserve its key claims, approach, "
    "and any apparent gaps or errors; discard low-signal filler. Do not judge "
    "or rewrite — only summarize."
)


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _summarize_prompt(task: str, answer: str) -> str:
    return (
        "Summarize the candidate answer below into a compact, faithful synopsis "
        "that preserves its key claims, approach, and any apparent gaps or "
        "errors — for comparing it against other candidate answers.\n\n"
        f"Task:\n{task}\n\nCandidate answer:\n{answer}"
    )


def _select_prompt(task: str, texts: Sequence[str]) -> str:
    body = "\n\n".join(
        f"Candidate {position}:\n{text}" for position, text in enumerate(texts, start=1)
    )
    return (
        "Several candidate answers to the same task are below. Pick the single "
        "best one.\n\n"
        f"Task:\n{task}\n\n"
        f"{body}\n\n"
        f"Reply with ONLY the number (1 to {len(texts)}) of the best candidate."
    )


def run_rtv(
    worker: Agent | Sequence[Agent],
    selector: Agent,
    task: str,
    *,
    rollouts: int = 8,
    group_size: int = 3,
    summarizer: Agent | None = None,
    worker_max_turns: int = 20,
    selector_max_turns: int = 4,
    summarizer_max_turns: int = 4,
    max_concurrency: int = 8,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Run N independent rollouts and select the best via a comparison tournament.

    `worker` is either one agent run `rollouts` times (diversity comes from a
    non-zero sampling temperature on the provider) or a sequence of distinct
    agents (then `rollouts` is ignored and the sequence length is used). Each
    rollout runs concurrently via `run_parallel`. When `summarizer` is given,
    each rollout is first compressed to a synopsis that the selector compares;
    otherwise the rollout's raw answer is compared. The tournament splits the
    survivors into groups of `group_size`, the `selector` picks one per group,
    and the round repeats until a single rollout remains — whose **full** answer
    is the result (selection is on summaries, the output is always the rollout).

    `steps` are, in order: every rollout, then every summary, then every
    selector comparison — a complete audit trail with each sub-run's `State`.
    """
    if isinstance(worker, Agent):
        workers: list[Agent] = [worker] * max(1, rollouts)
    else:
        workers = list(worker)
        if not workers:
            raise ValueError("run_rtv requires at least one worker")
    group_size = max(2, group_size)
    task_text = as_text(task)

    fan = run_parallel(
        workers,
        task_text,
        worker_max_turns=worker_max_turns,
        max_concurrency=max_concurrency,
        abort=abort,
    )
    rollout_steps = fan.steps

    summary_steps: list[StepResult] = []
    candidate_text: list[str] = []
    for rollout in rollout_steps:
        if summarizer is None or abort():
            candidate_text.append(rollout.output)
            continue
        step = run_agent(
            summarizer,
            _summarize_prompt(task_text, rollout.output),
            max_turns=summarizer_max_turns,
            abort=abort,
            role="summarizer",
        )
        summary_steps.append(step)
        candidate_text.append(step.output or rollout.output)

    selector_steps: list[StepResult] = []
    alive = list(range(len(rollout_steps)))
    while len(alive) > 1 and not abort():
        winners: list[int] = []
        for chunk in _chunks(alive, group_size):
            if len(chunk) == 1:
                winners.append(chunk[0])
                continue
            step = run_agent(
                selector,
                _select_prompt(task_text, [candidate_text[i] for i in chunk]),
                max_turns=selector_max_turns,
                abort=abort,
                role="selector",
            )
            selector_steps.append(step)
            local = pick_index(step.output, len(chunk))
            winners.append(chunk[local])
        alive = winners

    winner = alive[0] if alive else 0
    return WorkflowResult(
        output=rollout_steps[winner].output,
        steps=[*rollout_steps, *summary_steps, *selector_steps],
    )


def make_selector_agent(
    provider: LLMProvider,
    *,
    name: str = "selector",
    role: str = SELECTOR_ROLE,
    system_prompt: str = SELECTOR_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the tournament selector `Agent` for `run_rtv`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )


def make_summarizer_agent(
    provider: LLMProvider,
    *,
    name: str = "summarizer",
    role: str = SUMMARIZER_ROLE,
    system_prompt: str = SUMMARIZER_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the optional rollout-summarizer `Agent` for `run_rtv`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
