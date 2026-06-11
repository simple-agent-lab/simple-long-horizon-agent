"""Reflection (generator / critic loop).

A *generator* produces a draft; a *critic* reviews it; the generator revises
using that feedback. Repeat until the critic is satisfied or a round budget
is hit. This self-correction loop reliably lifts quality on tasks with a
clear notion of "good" (code, proofs, structured writing).

Generator and critic are ordinary agents driven by the core ReAct loop. The
workflow owns only the loop's *outer* structure (draft -> critique -> revise)
and the stop condition; each agent's own turns/tools are unchanged.

Stop condition: the critic is asked to emit an approval marker (default
"APPROVED") once the draft is good enough. `run_reflection` watches for that
marker in the critic's output and stops the loop when it appears.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AbortFlag, AgentTool

from .base import StepResult, WorkflowResult, as_text, never_abort, run_agent

DEFAULT_APPROVAL_MARKER = "APPROVED"

GENERATOR_ROLE = "Produce and then revise a solution to the task."
GENERATOR_SYSTEM_PROMPT = (
    "You are a generation agent. Produce the best solution you can to the "
    "given task. When you are given critic feedback on a previous draft, "
    "address every point and return the COMPLETE revised solution (not a "
    "diff and not just the changes)."
)

CRITIC_ROLE = "Critique a draft and approve it once it is good enough."
CRITIC_SYSTEM_PROMPT = (
    "You are a critic agent. Review the draft against the task and give "
    "specific, actionable feedback on what to fix or improve. Be concrete "
    f"and prioritize the most important issues. When the draft fully and "
    f"correctly satisfies the task, reply with the single word "
    f"{DEFAULT_APPROVAL_MARKER!r} on its own line and nothing else. Do not "
    f"use that word unless you are approving."
)


def is_approved(critic_output: str, marker: str = DEFAULT_APPROVAL_MARKER) -> bool:
    """True when the critic signalled approval via `marker` (case-insensitive)."""
    return marker.strip().lower() in critic_output.lower()


def _draft_prompt(task: str) -> str:
    return f"Produce a solution to the following task.\n\nTask:\n{task}"


def _critique_prompt(task: str, draft: str) -> str:
    return (
        "Review the draft solution against the task.\n\n"
        f"Task:\n{task}\n\n"
        f"Draft:\n{draft}"
    )


def _revise_prompt(task: str, draft: str, feedback: str) -> str:
    return (
        "Revise your draft to address the critic's feedback. Return the "
        "complete revised solution.\n\n"
        f"Task:\n{task}\n\n"
        f"Previous draft:\n{draft}\n\n"
        f"Critic feedback:\n{feedback}"
    )


def run_reflection(
    generator: Agent,
    critic: Agent,
    task: str,
    *,
    max_rounds: int = 3,
    approval_marker: str = DEFAULT_APPROVAL_MARKER,
    generator_max_turns: int = 20,
    critic_max_turns: int = 10,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Draft with `generator`, then critique/revise up to `max_rounds` times.

    Flow: generator drafts -> critic reviews. If the critic approves, stop
    and return the current draft. Otherwise the generator revises against the
    feedback and the next round critiques that revision. `max_rounds` bounds
    the number of critique/revise cycles, so the run always terminates.

    The output is the latest draft. `steps` records every agent run in order
    (draft, critique, revision, critique, …) for inspection.
    """
    task_text = as_text(task)
    steps: list[StepResult] = []

    draft_step = run_agent(
        generator,
        _draft_prompt(task_text),
        max_turns=generator_max_turns,
        abort=abort,
        role="generator",
    )
    steps.append(draft_step)
    draft = draft_step.output

    for _ in range(max(0, max_rounds)):
        if abort():
            break
        critique_step = run_agent(
            critic,
            _critique_prompt(task_text, draft),
            max_turns=critic_max_turns,
            abort=abort,
            role="critic",
        )
        steps.append(critique_step)
        if is_approved(critique_step.output, approval_marker):
            break
        if abort():
            break
        revise_step = run_agent(
            generator,
            _revise_prompt(task_text, draft, critique_step.output),
            max_turns=generator_max_turns,
            abort=abort,
            role="generator",
        )
        steps.append(revise_step)
        draft = revise_step.output

    return WorkflowResult(output=draft, steps=steps)


def make_generator_agent(
    provider: LLMProvider,
    *,
    name: str = "generator",
    role: str = GENERATOR_ROLE,
    system_prompt: str = GENERATOR_SYSTEM_PROMPT,
    tools: Sequence[AgentTool] = (),
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a generator `Agent` for the reflection loop."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=tools,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )


def make_critic_agent(
    provider: LLMProvider,
    *,
    name: str = "critic",
    role: str = CRITIC_ROLE,
    system_prompt: str = CRITIC_SYSTEM_PROMPT,
    tools: Sequence[AgentTool] = (),
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a critic `Agent`. Its system prompt must mention the marker that
    `run_reflection`'s `approval_marker` looks for (default "APPROVED")."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=tools,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )
