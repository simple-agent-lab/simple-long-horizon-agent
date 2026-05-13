"""Bash-use preset agent.

Usage::

    from simple_agent_lab.agents.bash import make_bash_use_agent, make_bash_tool

    agent = make_bash_use_agent(provider)
    runtime = AgentRuntime([agent], tools=[make_bash_tool(cwd=...)])

The preset is intentionally minimal — one step that calls bash, then
finalizes. Override `name` / `role` / `system_prompt` on
`make_bash_use_agent` to reuse the loop with a different prompt.
"""

from .agent import (
    BASH_AGENT_DEFAULT_NAME,
    BASH_AGENT_DEFAULT_ROLE,
    BASH_AGENT_SYSTEM_PROMPT,
    DEFAULT_BASH_DEMO_COMMAND,
    DEFAULT_BASH_DEMO_TASK,
    bash_agent_until_final,
    bash_task_for_command,
    make_bash_use_agent,
    run_bash_agent_demo,
)
from .tool import (
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
    "DEFAULT_BASH_DEMO_COMMAND",
    "DEFAULT_BASH_DEMO_TASK",
    "DEFAULT_BASH_MAX_OUTPUT_CHARS",
    "DEFAULT_BASH_TIMEOUT_SECONDS",
    "MAX_BASH_TIMEOUT_SECONDS",
    "bash_agent_until_final",
    "bash_execution_to_tool_result",
    "bash_task_for_command",
    "budget_text",
    "detect_blocked_sleep_pattern",
    "format_bash_observation",
    "interpret_command_result",
    "make_bash_tool",
    "make_bash_use_agent",
    "run_bash",
    "run_bash_agent_demo",
    "strip_empty_lines",
]
