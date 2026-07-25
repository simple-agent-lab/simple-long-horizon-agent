"""Parallelization (fan-out, then optional aggregate).

Run several agents at once, then optionally fold their answers into one. Two
common shapes both fit here:

- **Ensemble / voting** — several workers attempt the *same* task and an
  aggregator reconciles them (diverse takes, majority answer).
- **Sectioning / map-reduce** — each worker handles a *different* sub-task
  (pass `tasks=`) and the aggregator stitches the pieces together.

Each worker and the aggregator are ordinary agents driven by the core ReAct
loop. Workers run concurrently in a thread pool — every sub-run has its own
`State`, so there is no shared mutable loop state to race on. The pool here
parallelizes whole agent runs; it is the same idea as `core.dispatch_tool_calls`
running a turn's tool calls concurrently, one level up.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.tools import AbortFlag, AgentTool

from .base import (
    StepResult,
    WorkflowResult,
    as_text,
    make_role_agent,
    never_abort,
    run_agent,
)

AGGREGATOR_ROLE = "Synthesize several agents' answers into one final answer."
AGGREGATOR_SYSTEM_PROMPT = (
    "You are an aggregation agent. You are given a task and several candidate "
    "answers produced independently by other agents. Synthesize them into one "
    "best final answer: reconcile disagreements, keep what is correct, drop "
    "what is wrong, and do not merely concatenate them."
)


def _combine_outputs(steps: Sequence[StepResult]) -> str:
    """Default no-aggregator output: labelled concatenation of worker answers."""
    return "\n\n".join(f"## {step.name}\n{step.output}" for step in steps)


def _aggregate_prompt(task: str, steps: Sequence[StepResult]) -> str:
    answers = "\n\n".join(
        f"--- Answer from {step.name} ---\n{step.output}" for step in steps
    )
    return (
        "Synthesize the candidate answers below into one best final answer.\n\n"
        f"Task:\n{task}\n\n"
        f"{answers}"
    )


def run_parallel(
    workers: Sequence[Agent],
    task: str,
    *,
    aggregator: Agent | None = None,
    tasks: Sequence[str] | None = None,
    worker_max_turns: int = 20,
    aggregator_max_turns: int = 20,
    max_concurrency: int = 8,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Run `workers` concurrently, then optionally aggregate their outputs.

    By default every worker gets the same `task` (ensemble). Pass `tasks` —
    one entry per worker — to give each a different sub-task (map-reduce). If
    `aggregator` is given, it runs on all worker answers and its output is the
    result; otherwise the result is a labelled concatenation of the answers.

    Worker steps appear in `steps` in the original `workers` order regardless
    of completion order; the aggregator step (if any) is appended last.
    """
    worker_list = list(workers)
    if not worker_list:
        raise ValueError("run_parallel requires at least one worker")
    if tasks is not None and len(tasks) != len(worker_list):
        raise ValueError(
            f"tasks has {len(tasks)} entries but there are {len(worker_list)} workers"
        )

    task_text = as_text(task)
    worker_tasks = (
        [as_text(t) for t in tasks]
        if tasks is not None
        else [task_text] * len(worker_list)
    )

    workers_count = min(max_concurrency, len(worker_list))
    results: dict[int, StepResult] = {}
    with ThreadPoolExecutor(max_workers=workers_count) as pool:
        future_to_index = {
            pool.submit(
                run_agent,
                agent,
                worker_tasks[index],
                max_turns=worker_max_turns,
                abort=abort,
                role="worker",
            ): index
            for index, agent in enumerate(worker_list)
        }
        for future in future_to_index:
            index = future_to_index[future]
            results[index] = future.result()

    steps = [results[index] for index in range(len(worker_list))]

    if aggregator is None or abort():
        return WorkflowResult(output=_combine_outputs(steps), steps=steps)

    agg_step = run_agent(
        aggregator,
        _aggregate_prompt(task_text, steps),
        max_turns=aggregator_max_turns,
        abort=abort,
        role="aggregator",
    )
    return WorkflowResult(output=agg_step.output, steps=[*steps, agg_step])


def make_aggregator_agent(
    provider: LLMProvider,
    *,
    tools: Sequence[AgentTool] = (),
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the aggregator `Agent` for `run_parallel`."""
    return make_role_agent(
        provider,
        name="aggregator",
        role=AGGREGATOR_ROLE,
        system_prompt=AGGREGATOR_SYSTEM_PROMPT,
        tools=tools,
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
