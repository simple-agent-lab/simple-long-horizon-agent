"""Bash + task agent preset.

Builds a parent ``Agent`` with two tools:

- ``bash``: run a single focused command in the workspace.
- ``task``: delegate a sub-task to a bash worker (``subagent_type="explorer"``)
  whose own bash tool runs in the same workspace. The worker returns one
  short final summary, so heavy exploration does not flood the parent's
  context.

The worker (``explorer``) only has a bash tool — it cannot delegate
again. That keeps the sub-call shape simple and predictable, and avoids
unbounded recursion.

Usage::

    from simple_agent_lab.agents.bash_task import make_bash_task_agent

    agent = make_bash_task_agent(provider, cwd=workdir)
    state, events = agent.run(task)
    for _ in events: pass
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.agents.bash import BASH_AGENT_SYSTEM_PROMPT, make_bash_agent
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.task import task_tool


BASH_TASK_AGENT_DEFAULT_NAME = "bash_task_agent"
BASH_TASK_AGENT_DEFAULT_ROLE = (
    "Drive the task with bash for focused commands and delegate heavy "
    "exploration or multi-file reads to the bash worker via the task tool."
)
# Small addendum bolted onto the existing bash-agent system prompt so the
# parent inherits all of its phrasing about non-interactive flags, parallel
# reads, etc., and only learns the additional `task`-tool affordance.
BASH_TASK_EXPLORER_ADDENDUM = (
    "You also have a `task` tool that delegates a sub-task to a bash "
    'worker (`subagent_type="explorer"`); strongly prefer it for any '
    "investigation step (locating relevant code, mapping how a feature "
    "is used, reading a long file, tracing a failing test). Use `bash` "
    "directly only for the actual edits and the focused tests you run "
    "to verify them — not for exploration."
)
BASH_TASK_AGENT_SYSTEM_PROMPT = (
    BASH_AGENT_SYSTEM_PROMPT + "\n\n" + BASH_TASK_EXPLORER_ADDENDUM
)

EXPLORER_AGENT_DEFAULT_NAME = "explorer"
EXPLORER_AGENT_DEFAULT_ROLE = (
    "Run focused bash exploration in the workspace and return a short "
    "summary of what was found."
)
EXPLORER_AGENT_SYSTEM_PROMPT = (
    "You are a bash-using exploration sub-agent. Your job is to satisfy "
    "the delegated task by running bash commands (parallel reads when the "
    "steps are independent), then return a short final summary that the "
    "parent can use to plan its next edit. Prefer narrow tools (`grep -n`, "
    "`head`/`tail`, `sed -n 'A,Bp'`, `find`) over dumping whole files. "
    "Do NOT attempt edits — only investigate. Stop as soon as you have "
    "the evidence the task asks for."
)

DEFAULT_TASK_MAX_TURNS = 70


def make_bash_task_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_TASK_AGENT_DEFAULT_NAME,
    role: str = BASH_TASK_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_TASK_AGENT_SYSTEM_PROMPT,
    explorer_name: str = EXPLORER_AGENT_DEFAULT_NAME,
    explorer_role: str = EXPLORER_AGENT_DEFAULT_ROLE,
    explorer_system_prompt: str = EXPLORER_AGENT_SYSTEM_PROMPT,
    task_max_turns: int = DEFAULT_TASK_MAX_TURNS,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a parent ``Agent`` with bash + task(explorer) tools.

    Parent and explorer share ``cwd`` so a delegated investigation sees
    the same workspace state the parent's edits affect. Both run with
    the same provider — callers can pass distinct providers by composing
    the inner ``make_bash_agent`` directly if they want a cheaper model
    for exploration. ``request_extra`` flows to both so every model call
    carries the same per-request extras.
    """

    explorer = make_bash_agent(
        provider=provider,
        cwd=cwd,
        name=explorer_name,
        role=explorer_role,
        system_prompt=explorer_system_prompt,
        request_extra=request_extra,
    )
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=[
            make_bash_tool(cwd=cwd),
            task_tool([explorer], max_turns=task_max_turns),
        ],
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )
