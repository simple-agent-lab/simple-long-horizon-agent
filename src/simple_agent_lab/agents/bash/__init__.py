"""Bash-use preset agent.

Usage::

    from simple_agent_lab.agents.bash import make_bash_agent

    agent = make_bash_agent(provider, cwd=...)
    state, events = agent.run("list the files here")
    for _ in events: pass

The preset is intentionally minimal — one step that calls bash, then
finalizes. Override `name` / `role` / `system_prompt` on the factory to
reuse the loop with a different prompt.
"""

from .agent import (
    BASH_AGENT_DEFAULT_NAME,
    BASH_AGENT_DEFAULT_ROLE,
    BASH_AGENT_SYSTEM_PROMPT,
    make_bash_agent,
)
from simple_agent_lab.tools.bash import (
    BASH_TOOL_NAME,
    BashExecution,
    CommandInterpretation,
    DEFAULT_BASH_MAX_OUTPUT_CHARS,
    DEFAULT_BASH_TIMEOUT_SECONDS,
    MAX_BASH_TIMEOUT_SECONDS,
    bash_execution_to_tool_result,
    budget_text,
    detect_blocked_sleep_pattern,
    format_bash_observation,
    interpret_command_result,
    make_bash_tool,
    run_bash,
    strip_empty_lines,
)


__all__ = [
    "BASH_AGENT_DEFAULT_NAME",
    "BASH_AGENT_DEFAULT_ROLE",
    "BASH_AGENT_SYSTEM_PROMPT",
    "BASH_TOOL_NAME",
    "BashExecution",
    "CommandInterpretation",
    "DEFAULT_BASH_MAX_OUTPUT_CHARS",
    "DEFAULT_BASH_TIMEOUT_SECONDS",
    "MAX_BASH_TIMEOUT_SECONDS",
    "bash_execution_to_tool_result",
    "budget_text",
    "detect_blocked_sleep_pattern",
    "format_bash_observation",
    "interpret_command_result",
    "make_bash_agent",
    "make_bash_tool",
    "run_bash",
    "strip_empty_lines",
]
