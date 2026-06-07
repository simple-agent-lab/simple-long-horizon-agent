"""AgentBench: 39 tasks across 7 domains with 4-layer scoring.

Each task lives in data/tasks/{suite}/{task_name}/task.yaml containing:
  - name, id, suite, difficulty, mode, description
  - user_message: the agent-visible prompt
  - input_files: list of input files to copy into workspace
  - expected_outputs: patterns + validators (file-exists, content-contains,
    word-count-range, etc.)
  - scoring: layer weight overrides (layer0/1/2/3 weights)

Scoring is 4-layer:
  - Layer 0 (structural): file-exists, content-contains checks (done in evaluate)
  - Layer 1 (metrics): tool call count, planning ratio (host-side)
  - Layer 2 (behavioral): instruction adherence, tool appropriateness (host-side)
  - Layer 3 (output quality): completeness, accuracy, formatting (host-side)

The container evaluate hook handles L0 checks. L1-L3 are host-side.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from simple_agent_lab.evals.protocols import LaunchSpec


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Walk data/tasks/{suite}/{task_name}/task.yaml and load all tasks."""
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    tasks: list[dict[str, Any]] = []
    for task_yaml in sorted(tasks_dir.rglob("task.yaml")):
        try:
            with open(task_yaml, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue

        if not meta or not isinstance(meta, dict):
            continue

        task_dir = task_yaml.parent

        # Check for input files directory
        input_dir = task_dir / "inputs"
        input_files: list[str] = []
        if input_dir.exists():
            for f in input_dir.rglob("*"):
                if f.is_file():
                    input_files.append(f.name)

        tasks.append({
            "instance_id": meta.get("id", task_dir.name),
            "name": meta.get("name", ""),
            "suite": meta.get("suite", ""),
            "difficulty": meta.get("difficulty", "medium"),
            "mode": meta.get("mode", "sandboxed"),
            "description": meta.get("description", ""),
            "user_message": meta.get("user_message", ""),
            "input_file_names": [
                f.get("name", str(f)) if isinstance(f, dict) else str(f)
                for f in meta.get("input_files", [])
            ],
            "available_input_files": input_files,
            "expected_outputs": meta.get("expected_outputs", []),
            "expected_metrics": meta.get("expected_metrics", {}),
            "scoring_weights": meta.get("scoring", {}),
            "task_dir": str(task_dir.relative_to(data_dir)),
        })

    return tasks


class AgentBenchSuite:
    """Suite for AgentBench: 39 tasks with 4-layer scoring."""

    name = "agentbench"
    container_module = "simple_agent_lab.evals.suites.agentbench.container"

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
        """Agent-visible input: user_message + input file info."""
        return {
            "instance_id": instance["instance_id"],
            "user_message": instance["user_message"],
            "input_file_names": instance.get("input_file_names", []),
            "difficulty": instance.get("difficulty", "medium"),
            "task_dir": instance.get("task_dir", ""),
            "data_dir": str(self._data_dir),
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Gold scoring data for Layer 0 checks in the evaluate hook."""
        return {
            "instance_id": instance["instance_id"],
            "expected_outputs": instance.get("expected_outputs", []),
            "scoring_weights": instance.get("scoring_weights", {}),
        }
