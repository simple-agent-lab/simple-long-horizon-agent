from __future__ import annotations

from collections.abc import Mapping

from simple_agent_lab.evolution.surface import AgentSurface, SurfaceComponent


def default_harness_files() -> dict[str, str]:
    return {
        "agent_program.py": _agent_program_py(),
        "code_agent.yaml": _code_agent_yaml(),
        "systemprompt.md": _systemprompt_md(),
        "LongTermMEMORY.md": _long_term_memory_md(),
        "ShortTermMEMORY.md": _short_term_memory_md(),
        "tool_descriptions/bash.tool.md": _bash_tool_description_md(),
        "tools/bash.py": _bash_tool_py(),
        "middleware/README.md": _middleware_readme_md(),
        "skills/README.md": _skills_readme_md(),
        "sub_agents/README.md": _sub_agents_readme_md(),
    }


def ahe_harness_surface(*, artifact_key: str) -> AgentSurface:
    default_files = _with_harness_root(default_harness_files())
    return AgentSurface(
        id="ahe_harness_surface",
        name="AHE harness surface",
        description="An AHE-shaped harness package with explicit agent, memory, and tool files.",
        entrypoint="harness/agent_program.py:build_agent",
        default_files=default_files,
        artifact_key=artifact_key,
        components=(
            SurfaceComponent(
                id="agent_program",
                name="Agent program",
                description="The harness entrypoint and agent assembly.",
                paths=("harness/agent_program.py",),
                validators=("path_allowed", "python_syntax", "entrypoint_exists"),
            ),
            SurfaceComponent(
                id="system_prompt",
                name="System prompt",
                description="The base instruction file for the harness agent.",
                paths=("harness/systemprompt.md",),
            ),
            SurfaceComponent(
                id="tool_descriptions",
                name="Tool descriptions",
                description="Markdown descriptions of available harness tools.",
                paths=("harness/tool_descriptions/**",),
            ),
            SurfaceComponent(
                id="tool_implementations",
                name="Tool implementations",
                description="Python implementations of harness tools.",
                paths=("harness/tools/**",),
                validators=("path_allowed", "python_syntax"),
            ),
            SurfaceComponent(
                id="middleware",
                name="Middleware",
                description="Harness middleware notes and helpers.",
                paths=("harness/middleware/**",),
            ),
            SurfaceComponent(
                id="skills",
                name="Skills",
                description="Harness-local skill notes.",
                paths=("harness/skills/**",),
            ),
            SurfaceComponent(
                id="sub_agents",
                name="Sub agents",
                description="Harness-local sub-agent notes.",
                paths=("harness/sub_agents/**",),
            ),
            SurfaceComponent(
                id="long_term_memory",
                name="Long-term memory",
                description="Long- and short-term memory notes.",
                paths=("harness/LongTermMEMORY.md", "harness/ShortTermMEMORY.md"),
            ),
            SurfaceComponent(
                id="everything",
                name="Whole harness package",
                description="Unrestricted edits to the whole harness package.",
                paths=("harness/**",),
                validators=("path_allowed", "python_syntax", "entrypoint_exists"),
            ),
        ),
    )


def _with_harness_root(files: Mapping[str, str]) -> dict[str, str]:
    return {f"harness/{path}": text for path, text in files.items()}


def _agent_program_py() -> str:
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "from simple_agent_lab.agents.starter import make_bash_agent\n\n\n"
        "def build_agent(*, provider, cwd, base_system_prompt):\n"
        "    package_dir = Path(__file__).resolve().parent\n"
        "    system_prompt = (package_dir / 'systemprompt.md').read_text(encoding='utf-8').strip()\n"
        "    long_term_memory = (package_dir / 'LongTermMEMORY.md').read_text(encoding='utf-8').strip()\n"
        "    short_term_memory = (package_dir / 'ShortTermMEMORY.md').read_text(encoding='utf-8').strip()\n"
        "    parts = [part.strip() for part in (base_system_prompt, system_prompt, long_term_memory, short_term_memory) if part and part.strip()]\n"
        "    system_prompt = '\\n\\n'.join(parts)\n"
        "    return make_bash_agent(provider=provider, cwd=cwd, system_prompt=system_prompt)\n"
    )


def _code_agent_yaml() -> str:
    return (
        "name: ahe-harness\n"
        "entrypoint: harness/agent_program.py:build_agent\n"
        "tools:\n"
        "  - bash\n"
    )


def _systemprompt_md() -> str:
    return (
        "You are an AHE harness agent.\n"
        "Use bash for focused local work, keep changes small, and explain what you observed.\n"
    )


def _long_term_memory_md() -> str:
    return (
        "# Long Term Memory\n\n"
        "Keep durable facts here: stable conventions, accepted decisions, and useful reminders.\n"
    )


def _short_term_memory_md() -> str:
    return (
        "# Short Term Memory\n\n"
        "Keep temporary notes here for the current run.\n"
    )


def _bash_tool_description_md() -> str:
    return (
        "# bash\n\n"
        "Run a focused shell command in the harness workspace.\n"
    )


def _bash_tool_py() -> str:
    return (
        "from __future__ import annotations\n\n"
        "from simple_agent_lab.tools.bash import make_bash_tool\n\n\n"
        "def build_tool(*, cwd):\n"
        "    return make_bash_tool(cwd=cwd)\n"
    )


def _middleware_readme_md() -> str:
    return (
        "# Middleware\n\n"
        "Harness middleware lives here when the harness grows additional hooks.\n"
    )


def _skills_readme_md() -> str:
    return (
        "# Skills\n\n"
        "Harness-local skills live here.\n"
    )


def _sub_agents_readme_md() -> str:
    return (
        "# Sub Agents\n\n"
        "Harness-local sub-agent notes live here.\n"
    )
