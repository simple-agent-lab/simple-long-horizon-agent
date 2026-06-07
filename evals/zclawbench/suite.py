"""ZClawBench: 116 tasks scored by host-side LLM judge.

Data is a simple tasks.json mapping task_id -> {"category": "..."}.
The prompts are generated dynamically based on the task_id and category.

Since zclawbench has only metadata (task_id + category) and no embedded
prompts, the container half simply forwards the task_id to the agent.
Actual prompts are injected at runtime by the orchestrator or generated
from the category. Scoring is done host-side by an LLM judge (score_fn
in run_benches.py), NOT in the container.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Load tasks from tasks.json.

    The JSON structure is:
      {"total_tasks": 116, "total_records": 696, "models_per_task": 6,
       "tasks": {"zcb_001": {"category": "..."}, ...}}
    """
    tasks_file = data_dir / "tasks.json"
    if not tasks_file.exists():
        return []

    with open(tasks_file, encoding="utf-8") as f:
        data = json.load(f)

    raw_tasks = data.get("tasks", {})
    tasks = []
    for task_id, info in sorted(raw_tasks.items()):
        tasks.append({
            "instance_id": task_id,
            "category": info.get("category", "unknown"),
        })

    return tasks


class ZClawBenchSuite:
    """Suite for ZClawBench: 116 tasks scored by host-side LLM judge."""

    name = "zclawbench"
    container_module = "simple_agent_lab.evals.suites.zclawbench.container"

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
        """Agent-visible input: task_id and category for prompt generation."""
        return {
            "instance_id": instance["instance_id"],
            "category": instance["category"],
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        """No in-container scoring. Scoring is done host-side by LLM judge."""
        return None
