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
        "Create a self-contained research brief that demonstrates information "
        "search and synthesis for this category-derived task. Choose a concrete "
        "topic inside the category, state your assumptions, organize verified "
        "facts, and include a short source/evidence checklist. Save the report "
        "to output.md in the current workspace."
    ),
    "Office & Daily Tasks": (
        "Create a realistic office productivity deliverable for this "
        "category-derived task. Pick a concrete scenario, define the requested "
        "recipient/user need, and produce the final artifact plus a brief action "
        "checklist. Save everything to output.md in the current workspace."
    ),
    "Data Analysis": (
        "Design and complete a compact data-analysis task for this category. "
        "Use a small synthetic dataset if no input data is provided, show the "
        "calculation or analysis steps, summarize findings, and save the final "
        "analysis to output.md in the current workspace."
    ),
    "Development & Operations": (
        "Complete a concrete development/operations deliverable for this "
        "category-derived task. Choose a realistic maintenance, debugging, or "
        "deployment scenario, write a concise solution plan or script snippet, "
        "include verification steps, and save it to output.md in the current "
        "workspace."
    ),
    "Automation": (
        "Design an automation for a concrete workflow in this category. Include "
        "trigger, inputs, processing steps, outputs, error handling, and a small "
        "script or pseudocode block. Save the deliverable to output.md in the "
        "current workspace."
    ),
    "Security": (
        "Perform a concrete security review for this category-derived task. "
        "Pick a realistic asset or workflow, identify risks, prioritize findings, "
        "recommend mitigations, and save the report to output.md in the current "
        "workspace."
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
        f"Create a concrete deliverable for category '{category}'. "
        "Choose a realistic scenario, state assumptions, produce a useful final "
        "artifact, and save it to output.md in the current workspace.",
    )

    return (
        f"[Task: {task_id} | Category: {category}]\n\n"
        "This adapter currently has category metadata but no original task prompt. "
        "Do not ask the user for more details. Instead, make a reasonable, explicit "
        "assumption and complete the requested category-derived deliverable.\n\n"
        f"{prompt}"
    )


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
