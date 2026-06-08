"""Shared plumbing for multi-agent workflows.

A *workflow* here is orchestration **around** agents, not a new agent loop.
Every individual agent in a workflow is driven by the project's one ReAct
loop — `simple_agent_lab.core.run`, reached through `Agent.run(task)` — so a
"planner", a "critic" or a "worker" is just an ordinary `Agent` whose
internal turn/tool loop is exactly the same code path the bash agent uses.

This module owns the two things every workflow needs:

- `run_agent(...)` — run one agent to completion on a task and capture its
  final text plus the full `State` (messages + trace events). It calls
  `Agent.run` and drains the event generator; it never re-implements the
  turn loop.
- `StepResult` / `WorkflowResult` — small records so a workflow can return
  both its final answer and a per-step audit trail (each step keeps its
  `State`, so the whole run is inspectable / traceable after the fact).

Workflows pass information between agents by composing plain task strings
(e.g. "here is the plan, now execute it"); the heavier `context=` escape
hatch seeds extra messages directly into a sub-run when a workflow needs
the sub-agent to *see* something without it being phrased as the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.messages import (
    AssistantMessage,
    ContentInput,
    Message,
    normalize_content,
    text_of,
)
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag


def never_abort() -> bool:
    """Default `AbortFlag`: never abort. Shared so every entry point agrees."""
    return False


@dataclass
class StepResult:
    """One agent run inside a workflow.

    `output` is the agent's final answer text (see `final_output`). `state`
    is the complete run state — its `.messages` and `.events` make the step
    replayable and traceable, so a workflow result is a full audit trail and
    not just a string.
    """

    name: str
    output: str
    state: State
    role: str = ""
    task: str = ""


@dataclass
class WorkflowResult:
    """The outcome of a whole workflow: a final answer + per-step records."""

    output: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def states(self) -> list[State]:
        """Each step's `State`, in order — convenient for tracing the run."""
        return [step.state for step in self.steps]


def final_output(state: State, agent_name: str) -> str:
    """Extract `agent_name`'s answer text from a finished run.

    Prefers the terminal `final` message (the one the core loop stops on).
    Falls back to the agent's last assistant utterance when the run hit
    `max_turns` without emitting `final`, so a truncated step still yields
    usable text for the next stage rather than an empty string.

    Uses `text_of` (full content), not `message_text` (a 120-char preview):
    a step's output is fed verbatim into the next agent's task, so it must
    not be truncated.
    """
    for message in reversed(state.messages):
        if message.sender == agent_name and message.kind == "final":
            return text_of(message.content)
    for message in reversed(state.messages):
        if isinstance(message, AssistantMessage) and message.sender == agent_name:
            text = text_of(message.content)
            if text:
                return text
    return ""


def as_text(task: ContentInput) -> str:
    """Best-effort plain text for a task (string passes through unchanged)."""
    if isinstance(task, str):
        return task
    return text_of(normalize_content(task))


def run_agent(
    agent: Agent,
    task: ContentInput,
    *,
    max_turns: int = 10,
    abort: AbortFlag = never_abort,
    context: Sequence[Message] = (),
    role: str = "",
) -> StepResult:
    """Run one agent on `task` to completion via the core ReAct loop.

    This is the single bridge between workflows and `core.run`: it calls
    `Agent.run` (which seeds the task and returns the event generator) and
    then drains the generator to actually advance the loop. Aborting stops
    the drain early. The agent's own turn/tool handling is untouched — the
    workflow only decides *what task* to hand it and *what to do with the
    answer*.

    `context` messages are recorded onto the sub-run's `State` right after
    the task and before the first turn, mirroring how `task_tool` injects
    delegation context: they become visible to the agent without being part
    of the task string. Most workflows leave this empty and put everything
    in `task`.
    """
    state, events = agent.run(task, max_turns=max_turns, abort=abort)
    for message in context:
        state.record(message)
    for _ in events:
        if abort():
            break
    return StepResult(
        name=agent.name,
        role=role or agent.role,
        task=as_text(task),
        output=final_output(state, agent.name),
        state=state,
    )
