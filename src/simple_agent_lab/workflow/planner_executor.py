"""Planner / executor architecture.

A classic two-role split: a *planner* turns a task into an ordered plan,
then an *executor* carries the plan out. Separating "decide what to do"
from "do it" keeps each prompt focused and makes the plan an inspectable
artifact (it is captured as the first step's output).

Both roles are ordinary agents driven by the core ReAct loop. The planner
typically needs no tools (it just thinks); the executor usually carries the
real tools (bash, read, …) so it can act. `run_planner_executor` only wires
the plan from one into the task of the other.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.tools import AbortFlag, AgentTool

from .base import WorkflowResult, as_text, make_role_agent, never_abort, run_agent

PLANNER_ROLE = "Turn a task into a short, ordered, actionable plan."
PLANNER_SYSTEM_PROMPT = (
    "You are a planning agent. Given a task, produce a concise numbered plan "
    "of concrete steps that an executor agent can follow. Do not carry the "
    "task out yourself and do not write the final answer — only the plan. "
    "Keep steps small, ordered, and unambiguous; call out any assumptions."
)

EXECUTOR_ROLE = "Carry out a plan to accomplish the task and report the result."
EXECUTOR_SYSTEM_PROMPT = (
    "You are an execution agent. You are given a task and a plan produced for "
    "it. Work through the plan step by step, using your tools where needed. If "
    "a step turns out to be wrong or impossible, adapt sensibly and say so. "
    "When done, return a clear final answer that accomplishes the task."
)


def _planner_prompt(task: str) -> str:
    return f"Produce a plan for the following task.\n\nTask:\n{task}"


def _executor_prompt(task: str, plan: str) -> str:
    return (
        "Accomplish the task by following the plan below.\n\n"
        f"Task:\n{task}\n\n"
        f"Plan:\n{plan}"
    )


def run_planner_executor(
    planner: Agent,
    executor: Agent,
    task: str,
    *,
    planner_max_turns: int = 10,
    executor_max_turns: int = 20,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Plan the task with `planner`, then execute the plan with `executor`.

    Returns a `WorkflowResult` whose `output` is the executor's final answer
    and whose two `steps` are the plan and the execution — so the plan stays
    visible for inspection. For an iterative variant (plan -> execute ->
    critique -> replan), wrap this with `workflow.reflection`.
    """
    task_text = as_text(task)
    plan_step = run_agent(
        planner,
        _planner_prompt(task_text),
        max_turns=planner_max_turns,
        abort=abort,
        role="planner",
    )
    exec_step = run_agent(
        executor,
        _executor_prompt(task_text, plan_step.output),
        max_turns=executor_max_turns,
        abort=abort,
        role="executor",
    )
    return WorkflowResult(output=exec_step.output, steps=[plan_step, exec_step])


def make_planner_agent(
    provider: LLMProvider,
    *,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the planner `Agent` for `run_planner_executor` (no tools: it only plans)."""
    return make_role_agent(
        provider,
        name="planner",
        role=PLANNER_ROLE,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )


def make_executor_agent(
    provider: LLMProvider,
    *,
    tools: Sequence[AgentTool] = (),
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the executor `Agent`. Pass `tools=` (bash, read, ...) so it can act."""
    return make_role_agent(
        provider,
        name="executor",
        role=EXECUTOR_ROLE,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        tools=tools,
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
