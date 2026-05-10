"""Tiny bash-use agent demo built on the canonical runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .bash_tool import make_bash_tool
from .core import Agent, AgentRuntime, Event, State, make_llm_step, until_final
from .llm import Provider as LLMProvider


DEFAULT_BASH_DEMO_COMMAND = (
    "pwd && find src/simple_agent_lab -maxdepth 1 -type f -name '*.py' | sort"
)
DEFAULT_BASH_DEMO_TASK = f"Use bash to run command: `{DEFAULT_BASH_DEMO_COMMAND}`"
_FAKE_PROVIDER = LLMProvider(id="fake", api="fake", model="fake-model")


def make_bash_use_agent(provider: LLMProvider | None = None) -> Agent:
    """Create a minimal agent that can use the local bash tool."""

    return Agent(
        name="bash_agent",
        role="Use bash for one local command, then summarize the observation.",
        step=make_llm_step(
            provider or _FAKE_PROVIDER,
            system_prompt=(
                "You are a tiny bash-use agent. If the task names a command, "
                "call the bash tool once. After the tool_result, return a short final answer."
            ),
            target="user",
        ),
    )


bash_agent_until_final = until_final("bash_agent")


def bash_task_for_command(command: str) -> str:
    """Build the demo task text parsed by the deterministic fake adapter."""

    return f"Use bash to run command: `{command}`"


def run_bash_agent_demo(
    *,
    task: str | None = None,
    command: str | None = None,
    cwd: str | Path | None = None,
    provider: LLMProvider | None = None,
    on_event: Callable[[Event], None] | None = None,
) -> AgentRuntime:
    """Run the deterministic bash-use demo and return its runtime state."""

    resolved_task = task or DEFAULT_BASH_DEMO_TASK
    if command is not None:
        resolved_task = bash_task_for_command(command)

    runtime = AgentRuntime(
        [make_bash_use_agent(provider)],
        tools=[make_bash_tool(cwd=cwd)],
    )
    for event in runtime.prompt(
        resolved_task,
        target="bash_agent",
        next_agent=bash_agent_until_final,
    ):
        if on_event is not None:
            on_event(event)
    return runtime
