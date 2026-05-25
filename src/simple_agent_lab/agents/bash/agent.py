"""Tiny bash-use agent preset.

`make_bash_agent(provider, cwd=...)` returns a ready-to-run `Agent` with
the bash tool already attached. Callers drive it with `agent.run(task)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.core import Agent
from simple_agent_lab.llm.step import make_llm_step
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.tools.bash import make_bash_tool

BASH_AGENT_SYSTEM_PROMPT = (
    "You are a tiny bash-use agent. Use the bash tool to satisfy the task "
    "— parallel tool calls are fine when the steps are independent, and "
    "set `attach` on the call when you want to surface an image file. "
    "After the tool_result, return a short final answer."
)
BASH_AGENT_DEFAULT_ROLE = (
    "Use bash for local commands, then summarize what you observed."
)
BASH_AGENT_DEFAULT_NAME = "bash_agent"


def make_bash_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a bash-using `Agent` with the bash tool already bound.

    Consumers (eval suites, demos, custom flows) own provider choice so
    this preset stays independent of fake or live model policy.
    """

    return Agent(
        name=name,
        role=role,
        step=make_llm_step(
            provider,
            system_prompt=system_prompt,
            target="user",
            request_extra=request_extra,
        ),
        tools=[make_bash_tool(cwd=cwd)],
    )
