"""Sequential chain (prompt chaining).

The simplest multi-agent shape: run agents one after another, feeding each
stage the previous stage's output. Good when a task decomposes into fixed,
ordered sub-steps (e.g. outline -> draft -> polish, or extract -> transform
-> format) and every step benefits from a focused agent/prompt.

Each stage is a full `Agent` driven by the core ReAct loop (via
`run_agent`); the chain only decides what each stage receives as its task.
"""

from __future__ import annotations

from typing import Callable, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.tools import AbortFlag

from .base import StepResult, WorkflowResult, as_text, never_abort, run_agent

# Builds the next stage's task from (original_task, previous_output).
JoinFn = Callable[[str, str], str]


def default_join(task: str, previous_output: str) -> str:
    """Carry the original task forward alongside the previous step's output."""
    return (
        f"Task:\n{task}\n\n"
        "Output from the previous step — build on it, do not start over:\n"
        f"{previous_output}"
    )


def run_chain(
    agents: Sequence[Agent],
    task: str,
    *,
    max_turns: int = 10,
    abort: AbortFlag = never_abort,
    join: JoinFn = default_join,
) -> WorkflowResult:
    """Run `agents` in order, threading each one's output into the next.

    The first agent receives `task` verbatim. Every later agent receives
    `join(task, previous_output)` — by default the original task plus the
    prior stage's answer, so a stage keeps the goal in view instead of only
    seeing its predecessor's text. The workflow's output is the last stage's
    output.
    """
    agent_list = list(agents)
    if not agent_list:
        raise ValueError("run_chain requires at least one agent")

    steps: list[StepResult] = []
    previous_output = ""
    task_text = as_text(task)
    for index, agent in enumerate(agent_list):
        stage_task = task_text if index == 0 else join(task_text, previous_output)
        step = run_agent(agent, stage_task, max_turns=max_turns, abort=abort)
        steps.append(step)
        previous_output = step.output
        if abort():
            break

    return WorkflowResult(output=steps[-1].output, steps=steps)
