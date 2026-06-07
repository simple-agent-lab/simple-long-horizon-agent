"""ZClawBench container half: build task + extract result.

ZClawBench tasks have only task_id and category. The agent receives a
generic prompt to demonstrate its capability in the given category.
Scoring is done host-side by an LLM judge, not in the container.

Overall logic:
1. build_task: generates a category-aware prompt for the agent
2. extract_result: captures the agent's final response from the trajectory
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


# Category-specific prompt templates for tasks that lack embedded prompts.
# These are fallback prompts when no external prompt source is available.
_CATEGORY_PROMPTS: dict[str, str] = {
    "Information Search & Gathering": (
        "You need to find and organize specific information. "
        "Search for relevant data about the given topic, verify accuracy, "
        "and compile a structured report of your findings. "
        "Save your findings to workspace/output.md."
    ),
    "Office & Daily Tasks": (
        "Complete the following office or daily productivity task. "
        "Follow the instructions precisely and save all output files "
        "to the workspace directory."
    ),
    "Data Analysis": (
        "Analyze the provided data to extract meaningful insights. "
        "Perform statistical analysis, create visualizations if needed, "
        "and write a summary report to workspace/analysis.md."
    ),
    "Development & Operations": (
        "Complete the following development or operations task. "
        "Write clean, well-documented code. Test your solution "
        "and save all files to the workspace directory."
    ),
    "Automation": (
        "Automate the described workflow or process. "
        "Create scripts or configurations that accomplish the task "
        "and save them to the workspace directory."
    ),
    "Security": (
        "Perform the security analysis or task as described. "
        "Document your findings and recommendations in workspace/report.md."
    ),
}


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="zclawbench_agent",
        role="General-purpose agent handling diverse tasks.",
        system_prompt=(
            "You are a capable agent that can handle diverse tasks across "
            "multiple categories including information gathering, data analysis, "
            "development, automation, and security. Follow instructions precisely "
            "and produce complete, well-structured output."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the agent-visible task prompt.

    Uses the category to generate a relevant prompt template.
    The orchestrator may override this with an externally injected prompt.
    """
    workspace = Path(workdir)
    workspace.mkdir(parents=True, exist_ok=True)

    task_id = instance.get("instance_id", "unknown")
    category = instance.get("category", "")

    # Use category-specific prompt, or generic fallback
    prompt = _CATEGORY_PROMPTS.get(
        category,
        f"Complete the task for category '{category}'. "
        f"Save your output to the workspace directory.",
    )

    return f"[Task: {task_id} | Category: {category}]\n\n{prompt}"


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """List workspace files produced by the agent."""
    files = []
    for f in sorted(workspace.rglob("*")):
        if f.is_file() and not f.name.startswith("_"):
            files.append(str(f.relative_to(workspace)))

    return {
        "instance_id": instance.get("instance_id", ""),
        "workspace_files": files,
        "status": "completed",
    }
