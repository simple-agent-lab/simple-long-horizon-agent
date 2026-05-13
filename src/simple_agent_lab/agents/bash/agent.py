"""Tiny bash-use agent — the first preset shipped on the lab runtime.

`make_bash_use_agent(provider)` builds a single-step agent that can call
the bash tool defined alongside in `tool.py`. The factory accepts
`name` / `role` / `system_prompt` overrides so a downstream consumer
(eval suites, custom flows) can reuse the same wiring without forking
the prompt.

`run_bash_agent_demo(...)` is the one-shot helper used by the demo
script — it spins up an `AgentRuntime`, hands it a bash command or a
free-form task, and runs to completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from simple_agent_lab.core import (
    Agent,
    AgentRuntime,
    Event,
    State,
    make_llm_step,
    until_final,
)
from simple_agent_lab.llm import Provider as LLMProvider

from .tool import make_bash_tool


DEFAULT_BASH_DEMO_COMMAND = (
    "pwd && find src/simple_agent_lab -maxdepth 1 -type f -name '*.py' | sort"
)
DEFAULT_BASH_DEMO_TASK = f"Use bash to run command: `{DEFAULT_BASH_DEMO_COMMAND}`"
BASH_AGENT_SYSTEM_PROMPT = (
    "You are a tiny bash-use agent. Use the bash tool to satisfy the task "
    "— parallel tool calls are fine when the steps are independent, and "
    "set `attach` on the call when you want to surface an image file. "
    "After the tool_result, return a short final answer."
)
BASH_AGENT_DEFAULT_ROLE = "Use bash for local commands, then summarize what you observed."
BASH_AGENT_DEFAULT_NAME = "bash_agent"

_FAKE_PROVIDER = LLMProvider(id="fake", api="fake", model="fake-model")


def make_bash_use_agent(
    provider: LLMProvider | None = None,
    *,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
) -> Agent:
    """Build a single-step bash-using agent.

    The defaults match the demo. Consumers (eval suites, custom flows)
    that want the same loop but a different prompt or routing name
    override `name` / `role` / `system_prompt`.
    """

    return Agent(
        name=name,
        role=role,
        step=make_llm_step(
            provider or _FAKE_PROVIDER,
            system_prompt=system_prompt,
            target="user",
        ),
    )


bash_agent_until_final = until_final(BASH_AGENT_DEFAULT_NAME)


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
    """Run the bash-use demo and return its runtime state."""

    resolved_task = task or DEFAULT_BASH_DEMO_TASK
    if command is not None:
        resolved_task = bash_task_for_command(command)

    runtime = AgentRuntime(
        [make_bash_use_agent(provider)],
        tools=[make_bash_tool(cwd=cwd)],
    )
    for event in runtime.prompt(
        resolved_task,
        target=BASH_AGENT_DEFAULT_NAME,
        next_agent=bash_agent_until_final,
    ):
        if on_event is not None:
            on_event(event)
    return runtime
