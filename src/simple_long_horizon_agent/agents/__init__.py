"""Agents that ship with simple-long-horizon-agent.

All agent kinds are built from one consolidated starter (`starter.py`) rather
than per-kind subfolders. A single :class:`AgentSession` opens any
resource-bearing :class:`Toolset` (e.g. an MCP server), assembles the tool
list, builds an ``Agent`` on the core runtime, and dispatches ``run`` to the
plain or skills loop. One composable `agent_session()` front door covers the
current kinds:

    from simple_long_horizon_agent.agents import agent_session

    with agent_session(provider, cwd=workdir, skills=True) as session:
        state, events = session.run(task)
        for _ in events:
            pass

Capabilities that own no live resource are built as a plain ``Agent`` via
``make_agent`` (general) or the named factories ``make_bash_agent`` /
``make_bash_task_agent`` / ``make_skill_agent``. Skills count as resource-free:
``make_skill_agent`` installs a skills state initializer on the agent, so a bare
``agent.run(task)`` is skills-aware with no session. Only MCP (a live
connection) needs ``agent_session``/``mcp_session``. These factories are not
auto-imported by the top-level ``simple_long_horizon_agent`` namespace — importing them
here keeps that surface focused on the protocol and runtime.
"""

from .flavors import (
    build_flavor_agent,
    make_workflow_runner_for_flavor,
)
from .starter import (
    BASH_AGENT_DEFAULT_NAME,
    BASH_AGENT_DEFAULT_ROLE,
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_DEFAULT_NAME,
    BASH_TASK_AGENT_DEFAULT_ROLE,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_ADDENDUM,
    DEFAULT_AGENT_NAME,
    DEFAULT_TASK_MAX_TURNS,
    DEFAULT_TASK_SOFT_TURN_LIMIT,
    GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
    GENERAL_PURPOSE_AGENT_DEFAULT_ROLE,
    GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT,
    MCP_ADDENDUM,
    SKILLS_ADDENDUM,
    AgentSession,
    SkillConfig,
    agent_session,
    compose_agent_system_prompt,
    make_agent,
    make_bash_agent,
    make_bash_task_agent,
    make_skill_agent,
    mcp_session,
)
from .toolsets import MCPToolset, Toolset

__all__ = [
    "AgentSession",
    "SkillConfig",
    "Toolset",
    "MCPToolset",
    "build_flavor_agent",
    "make_workflow_runner_for_flavor",
    "agent_session",
    "mcp_session",
    "compose_agent_system_prompt",
    "DEFAULT_AGENT_NAME",
    "SKILLS_ADDENDUM",
    "MCP_ADDENDUM",
    "make_agent",
    "make_skill_agent",
    "make_bash_agent",
    "make_bash_task_agent",
    "BASH_AGENT_SYSTEM_PROMPT",
    "BASH_AGENT_DEFAULT_ROLE",
    "BASH_AGENT_DEFAULT_NAME",
    "BASH_TASK_AGENT_SYSTEM_PROMPT",
    "BASH_TASK_AGENT_DEFAULT_ROLE",
    "BASH_TASK_AGENT_DEFAULT_NAME",
    "BASH_TASK_ADDENDUM",
    "GENERAL_PURPOSE_AGENT_DEFAULT_NAME",
    "GENERAL_PURPOSE_AGENT_DEFAULT_ROLE",
    "GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT",
    "DEFAULT_TASK_MAX_TURNS",
    "DEFAULT_TASK_SOFT_TURN_LIMIT",
]
