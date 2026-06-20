"""Shared plumbing for multi-agent workflows.

This module owns the two things every workflow needs:

- `run_agent(...)` — run one agent to completion on a task and capture its
  final text plus the full `State` (messages + trace events).
- `StepResult` / `WorkflowResult` — small records so a workflow can return
  both its final answer and a per-step audit trail (each step keeps its
  `State`, so the whole run is inspectable / traceable after the fact).

Workflows pass information between agents by composing plain task strings
(e.g. "here is the plan, now execute it"); the heavier `context=` escape
hatch seeds extra messages directly into a sub-run when a workflow needs
the sub-agent to *see* something without it being phrased as the task.
"""

from __future__ import annotations

import json
import re
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


def pick_index(text: str, n: int, *, default: int = 0) -> int:
    """Resolve a judge/selector's free text to a 0-based choice in ``[0, n)``.

    The selection seam shared by workflows that ask an agent to *pick the best
    of N candidates* (the tournament selector here, and future value-guided
    picks). Candidates are presented to the model 1-based ("Candidate 1..N"),
    so this parses a 1-based answer and returns it 0-based. Mirrors the tolerant
    parsing of `routing.select_route` / `goal_checks._parse_judge_json`: try a
    JSON object (`{"best": k}`) first, then the first in-range integer, and fall
    back to `default` when nothing usable is found — never raise on bad model
    output.
    """
    if n <= 0:
        raise ValueError("pick_index requires n >= 1")
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for key in ("best", "choice", "winner", "index", "answer"):
            if key in obj:
                try:
                    k = int(obj[key])
                except (TypeError, ValueError):
                    continue
                if 1 <= k <= n:
                    return k - 1
    for token in re.findall(r"\d+", text):
        k = int(token)
        if 1 <= k <= n:
            return k - 1
    return default


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


def resume_agent(
    agent: Agent,
    state: State,
    followup: ContentInput,
    *,
    max_turns: int = 10,
    abort: AbortFlag = never_abort,
    role: str = "",
) -> StepResult:
    """Continue an existing `state` with `followup` via the core ReAct loop.

    The `run_agent` counterpart for *continuation*: instead of seeding a fresh
    `State`, it appends `followup` to `state` and drives the loop, so messages
    and trace events keep accumulating on the same conversation. This is the
    seam tree search uses to grow a node — fork a parent `State` (`fork_state`),
    then `resume_agent` the fork independently.
    """
    state, events = agent.resume(state, followup, max_turns=max_turns, abort=abort)
    for _ in events:
        if abort():
            break
    return StepResult(
        name=agent.name,
        role=role or agent.role,
        task=as_text(followup),
        output=final_output(state, agent.name),
        state=state,
    )


def fork_state(state: State) -> State:
    """Return an independent copy of `state` that can be resumed in isolation.

    `State` is event-sourced (`events` + a `snapshot` replayed from them) and
    its `Message`/`Event` records are frozen, so a fork is just a fresh `State`
    carrying a shallow copy of the event list with its snapshot rebuilt — no
    deepcopy of immutable payloads. Resuming the fork appends new events to the
    copy only, leaving the parent untouched, which is what lets tree search
    branch one conversation into several without changing `core`.
    """
    forked = State(task=state.task, data=dict(state.data))
    forked.events = list(state.events)
    forked.rebuild_snapshot()
    return forked


def emitted_final(state: State, agent_name: str) -> bool:
    """True iff `agent_name` produced a terminal `final` message in `state`.

    Distinguishes a node that *finished* (the loop stopped on `final`) from one
    truncated at `max_turns` — tree search needs that to mark terminal leaves.
    """
    return any(
        message.sender == agent_name and message.kind == "final"
        for message in state.messages
    )
