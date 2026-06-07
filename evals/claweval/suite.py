"""ClawEval: 300 tasks with mock services + LLM judge scoring.

Each task lives in data/tasks/{task_id}/ containing:
  - task.yaml: task_id, task_name, category, prompt (text + language),
    tools, user_agent config, scoring_components, judge_rubric,
    reference_solution, primary_dimensions, optional services config
  - grader.py: custom grader class (optional, for specialized scoring)

Task categories include: user_agent (multi-turn with simulated user),
"what" (general Q&A), tool-using tasks, and more.

Scoring: host-side LLM judge. The judge_rubric in task.yaml provides
the rubric for LLM evaluation. Scoring components specify weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from simple_agent_lab.evals.protocols import LaunchSpec


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Walk data/tasks/{task_id}/task.yaml and load all tasks."""
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    tasks: list[dict[str, Any]] = []
    for task_yaml in sorted(tasks_dir.glob("*/task.yaml")):
        try:
            with open(task_yaml, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue

        if not meta or not isinstance(meta, dict):
            continue

        task_dir = task_yaml.parent
        task_id = meta.get("task_id", task_dir.name)

        # Read grader.py if it exists
        grader_code = ""
        grader_path = task_dir / "grader.py"
        if grader_path.exists():
            grader_code = grader_path.read_text(encoding="utf-8")

        # Extract prompt
        prompt = meta.get("prompt", {})
        if isinstance(prompt, dict):
            prompt_text = prompt.get("text", "")
            prompt_language = prompt.get("language", "en")
        else:
            prompt_text = str(prompt)
            prompt_language = "en"

        # Extract user_agent config
        user_agent = meta.get("user_agent", {})
        has_user_agent = user_agent.get("enabled", False)

        # Extract scoring components
        scoring_components = meta.get("scoring_components", [])

        # Extract services config
        services = meta.get("services", [])

        tasks.append({
            "instance_id": task_id,
            "task_name": meta.get("task_name", ""),
            "category": meta.get("category", "general"),
            "difficulty": meta.get("difficulty", "medium"),
            "tags": meta.get("tags", []),
            "prompt_text": prompt_text,
            "prompt_language": prompt_language,
            "tools": meta.get("tools", []),
            "tool_endpoints": meta.get("tool_endpoints", []),
            "user_agent_enabled": has_user_agent,
            "user_agent_persona": user_agent.get("persona", ""),
            "user_agent_max_rounds": user_agent.get("max_rounds", 5),
            "user_agent_system_prompt_suffix": user_agent.get("system_prompt_suffix", ""),
            "environment": meta.get("environment", {}),
            "scoring_components": scoring_components,
            "judge_rubric": meta.get("judge_rubric", ""),
            "reference_solution": meta.get("reference_solution", ""),
            "primary_dimensions": meta.get("primary_dimensions", []),
            "services": services,
            "grader_code": grader_code,
        })

    return tasks


class ClawEvalSuite:
    """Suite for ClawEval: 300 tasks with mock services + LLM judge scoring."""

    name = "claweval"
    container_module = "simple_agent_lab.evals.suites.claweval.container"

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        image: str = "clawbase-sal:v1",
    ) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self._data_dir = Path(data_dir)
        self._image = image
        self._tasks = load_tasks(self._data_dir)

    def load_instances(self) -> list[dict[str, Any]]:
        return list(self._tasks)

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image=self._image, workdir="/workspace")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Agent-visible input: prompt text + tool/user_agent config."""
        return {
            "instance_id": instance["instance_id"],
            "prompt_text": instance["prompt_text"],
            "prompt_language": instance.get("prompt_language", "en"),
            "tools": instance.get("tools", []),
            "user_agent_enabled": instance.get("user_agent_enabled", False),
            "user_agent_max_rounds": instance.get("user_agent_max_rounds", 5),
            "environment": instance.get("environment", {}),
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        """Gold scoring data for host-side LLM judge.

        Returns rubric, scoring components, and reference solution
        so the host-side judge can evaluate the agent's output.
        """
        return {
            "instance_id": instance["instance_id"],
            "scoring_components": instance.get("scoring_components", []),
            "judge_rubric": instance.get("judge_rubric", ""),
            "reference_solution": instance.get("reference_solution", ""),
            "user_agent_persona": instance.get("user_agent_persona", ""),
            "category": instance.get("category", ""),
            "difficulty": instance.get("difficulty", ""),
        }
