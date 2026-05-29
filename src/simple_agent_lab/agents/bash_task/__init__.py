"""Bash + task delegation preset agent.

Usage::

    from simple_agent_lab.agents.bash_task import make_bash_task_agent

    agent = make_bash_task_agent(provider, cwd=workdir)
    state, events = agent.run("solve this bug")
    for _ in events: pass

The preset wires a parent agent with two tools — direct ``bash`` for
focused commands and ``task`` to delegate heavier exploration to a bash
worker sub-agent (``explorer``). The worker shares the parent's ``cwd``
so its findings stay consistent with the parent's edits.
"""

from .agent import (
    BASH_TASK_AGENT_DEFAULT_NAME,
    BASH_TASK_AGENT_DEFAULT_ROLE,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_EXPLORER_ADDENDUM,
    DEFAULT_TASK_MAX_TURNS,
    EXPLORER_AGENT_DEFAULT_NAME,
    EXPLORER_AGENT_DEFAULT_ROLE,
    EXPLORER_AGENT_SYSTEM_PROMPT,
    make_bash_task_agent,
)


__all__ = [
    "BASH_TASK_AGENT_DEFAULT_NAME",
    "BASH_TASK_AGENT_DEFAULT_ROLE",
    "BASH_TASK_AGENT_SYSTEM_PROMPT",
    "BASH_TASK_EXPLORER_ADDENDUM",
    "DEFAULT_TASK_MAX_TURNS",
    "EXPLORER_AGENT_DEFAULT_NAME",
    "EXPLORER_AGENT_DEFAULT_ROLE",
    "EXPLORER_AGENT_SYSTEM_PROMPT",
    "make_bash_task_agent",
]
