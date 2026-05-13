"""Tiny bash-use agent preset.

`make_bash_agent_runtime(provider, cwd=...)` builds the reusable preset:
one bash-use agent plus the bash tool bound through `AgentRuntime`.
`make_bash_use_agent(provider)` remains the lower-level agent factory for
callers that want to compose their own runtime.
"""

from __future__ import annotations

from pathlib import Path

from simple_agent_lab.core import (
    Agent,
    make_llm_step,
    until_final,
)
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.runtime import AgentRuntime
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


def make_bash_use_agent(
    provider: LLMProvider,
    *,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
) -> Agent:
    """Build a single-step bash-using agent.

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
        ),
    )


def make_bash_agent_runtime(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
) -> AgentRuntime:
    """Build the bash-use runtime with its executable bash tool bound."""

    return AgentRuntime(
        [
            make_bash_use_agent(
                provider,
                name=name,
                role=role,
                system_prompt=system_prompt,
            )
        ],
        tools=[make_bash_tool(cwd=cwd)],
    )


bash_agent_until_final = until_final(BASH_AGENT_DEFAULT_NAME)
