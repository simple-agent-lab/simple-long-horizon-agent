"""Skill-using agent preset.

`make_skill_agent(provider, cwd=...)` returns an `Agent` carrying the two
tools the agent-skills workflow needs: `bash` (run a skill's scripts and
the focused commands that verify them) and `read` (open a skill's
`SKILL.md` and the references it points to).

The preset owns only the *agent construction* — the bash+read bundle plus a
skill-oriented prompt. Skill *delivery* stays the caller's choice, matching
the two existing call sites:

- Interactive: drive the returned agent through
  `simple_agent_lab.skills.run_with_skills(agent, task)`, which records the
  skills menu (and any `/mention` bodies) as messages before the task.
- Benchmark: pass a system prompt built with
  `simple_agent_lab.skills.system_prompt_with_skills(...)` so the menu rides
  the system prompt (the generic `agent.run` loop has no per-turn seam to
  record a menu into).

Either way the agent itself is the same self-contained bash+read value, so
the construction lives here once instead of being hand-rolled per caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool

SKILL_AGENT_SYSTEM_PROMPT = (
    "You are a capable software agent with a bash tool and a read tool. Use "
    "the available skills when they fit the task: read a skill's SKILL.md, "
    "then run its scripts via bash. Work from evidence and verify your result."
)
SKILL_AGENT_DEFAULT_ROLE = (
    "Use bash and read with the available skills, then verify your result."
)
SKILL_AGENT_DEFAULT_NAME = "skill_agent"


def make_skill_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = SKILL_AGENT_DEFAULT_NAME,
    role: str = SKILL_AGENT_DEFAULT_ROLE,
    system_prompt: str = SKILL_AGENT_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a skill-using `Agent` with the bash + read tools already bound.

    Consumers own provider choice and skill delivery (see the module
    docstring): pass a `system_prompt` built with `system_prompt_with_skills`
    to bake the menu in, or drive the returned agent through `run_with_skills`
    to inject it at run time.
    """
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=[make_bash_tool(cwd=cwd), make_read_tool(cwd=cwd)],
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )
