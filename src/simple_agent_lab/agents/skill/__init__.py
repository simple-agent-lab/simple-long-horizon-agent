"""Skill-using preset agent.

Usage::

    from simple_agent_lab.agents.skill import make_skill_agent
    from simple_agent_lab.skills import run_with_skills

    agent = make_skill_agent(provider, cwd=workdir)
    state, events = run_with_skills(agent, "/pdf fill this form")
    for _ in events: pass

The preset wires a `bash` + `read` agent — the shape the agent-skills
workflow needs (read a `SKILL.md`, run its scripts). Skill delivery is the
caller's choice: drive it with `run_with_skills` (interactive, menu injected
at run time) or pass a `system_prompt_with_skills(...)` prompt (benchmark,
menu baked into the system prompt).
"""

from .agent import (
    SKILL_AGENT_DEFAULT_NAME,
    SKILL_AGENT_DEFAULT_ROLE,
    SKILL_AGENT_SYSTEM_PROMPT,
    make_skill_agent,
)


__all__ = [
    "SKILL_AGENT_DEFAULT_NAME",
    "SKILL_AGENT_DEFAULT_ROLE",
    "SKILL_AGENT_SYSTEM_PROMPT",
    "make_skill_agent",
]
