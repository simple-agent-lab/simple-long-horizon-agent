"""Preset agents that ship with simple-agent-lab.

All agent kinds are built from one consolidated starter (`starter.py`) rather
than per-kind subfolders. A single :class:`AgentSession` opens any
resource-bearing :class:`Toolset` (e.g. an MCP server), assembles the tool
list, builds an ``Agent`` on the core runtime, and dispatches ``run`` to the
plain or skills loop. Four thin presets cover the current kinds:

    from simple_agent_lab.agents import bash_session, mcp_session

    with bash_session(provider, cwd=workdir) as session:
        state, events = session.run(task)
        for _ in events:
            pass

Resource-free kinds can still be built as a plain ``Agent`` via the
back-compat ``make_bash_agent`` / ``make_bash_task_agent`` factories. Presets
are not auto-imported by the top-level ``simple_agent_lab`` namespace —
importing them here keeps that surface focused on the protocol and runtime.
"""

from .starter import (
    BASH_AGENT_DEFAULT_NAME,
    BASH_AGENT_DEFAULT_ROLE,
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_DEFAULT_NAME,
    BASH_TASK_AGENT_DEFAULT_ROLE,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_EXPLORER_ADDENDUM,
    DEFAULT_TASK_MAX_TURNS,
    EXPLORER_AGENT_DEFAULT_NAME,
    EXPLORER_AGENT_DEFAULT_ROLE,
    EXPLORER_AGENT_SYSTEM_PROMPT,
    MCP_AGENT_DEFAULT_NAME,
    MCP_AGENT_DEFAULT_ROLE,
    SKILL_AGENT_DEFAULT_NAME,
    SKILL_AGENT_DEFAULT_ROLE,
    AgentSession,
    SkillConfig,
    bash_session,
    bash_task_session,
    make_bash_agent,
    make_bash_task_agent,
    mcp_session,
    skill_session,
)
from .toolsets import MCPToolset, Toolset

__all__ = [
    "AgentSession",
    "SkillConfig",
    "Toolset",
    "MCPToolset",
    "bash_session",
    "bash_task_session",
    "skill_session",
    "mcp_session",
    "make_bash_agent",
    "make_bash_task_agent",
    "BASH_AGENT_SYSTEM_PROMPT",
    "BASH_AGENT_DEFAULT_ROLE",
    "BASH_AGENT_DEFAULT_NAME",
    "BASH_TASK_AGENT_SYSTEM_PROMPT",
    "BASH_TASK_AGENT_DEFAULT_ROLE",
    "BASH_TASK_AGENT_DEFAULT_NAME",
    "BASH_TASK_EXPLORER_ADDENDUM",
    "EXPLORER_AGENT_DEFAULT_NAME",
    "EXPLORER_AGENT_DEFAULT_ROLE",
    "EXPLORER_AGENT_SYSTEM_PROMPT",
    "DEFAULT_TASK_MAX_TURNS",
    "SKILL_AGENT_DEFAULT_NAME",
    "SKILL_AGENT_DEFAULT_ROLE",
    "MCP_AGENT_DEFAULT_NAME",
    "MCP_AGENT_DEFAULT_ROLE",
]
