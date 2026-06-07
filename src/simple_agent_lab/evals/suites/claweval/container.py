"""ClawEval container half: build task + extract result.

ClawEval tasks include multi-turn user_agent simulations, tool-use
scenarios, and general Q&A. The agent receives the prompt and works
in the workspace. Scoring is done host-side by an LLM judge using
the judge_rubric and scoring_components from the task.

Overall logic:
1. build_task: returns the prompt text, adapting for user_agent tasks
   (appending system_prompt_suffix if multi-turn)
2. extract_result: captures workspace files and signals completion
3. No evaluate hook - scoring is host-side via LLM judge
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="claweval_agent",
        role="Expert agent handling diverse multi-turn and tool-use tasks.",
        system_prompt=(
            "You are a knowledgeable, helpful assistant. Answer questions "
            "accurately and thoroughly. When tools are available, use them "
            "appropriately. For multi-turn conversations, be patient and "
            "ask clarifying questions when needed. Provide complete, "
            "well-structured responses."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the agent-visible task from the instance data.

    For user_agent tasks (multi-turn), appends the system_prompt_suffix
    which instructs the agent to proactively ask clarifying questions.
    For regular tasks, simply returns the prompt text.
    """
    workspace = Path(workdir)
    workspace.mkdir(parents=True, exist_ok=True)

    prompt_text = instance.get("prompt_text", "")
    user_agent_enabled = instance.get("user_agent_enabled", False)
    system_prompt_suffix = instance.get(
        "user_agent_system_prompt_suffix", ""
    )

    # For multi-turn user_agent tasks, include the suffix guidance
    if user_agent_enabled and system_prompt_suffix:
        return f"{prompt_text}\n\n---\nGuidance: {system_prompt_suffix}"

    return str(prompt_text)


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """List workspace files and return completion status.

    For ClawEval, the primary artifact is the conversation trajectory
    (scored host-side by LLM judge). We also capture any files the
    agent created in the workspace.
    """
    files = []
    for f in sorted(workspace.rglob("*")):
        if f.is_file() and not f.name.startswith("_"):
            files.append(str(f.relative_to(workspace)))

    return {
        "instance_id": instance.get("instance_id", ""),
        "workspace_files": files,
        "status": "completed",
        "category": instance.get("category", ""),
        "user_agent_enabled": instance.get("user_agent_enabled", False),
    }
