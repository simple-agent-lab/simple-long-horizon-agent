"""SkillsBench: 88 tasks testing coding/algorithm skills with pytest scoring.

Each task lives under data/tasks/{task_slug}/ containing:
  - task.toml: version, metadata (difficulty, category, tags), verifier timeout, agent timeout
  - instruction.md: the agent-visible prompt describing the problem
  - environment/: Dockerfile + data files + skills/ (seeded into container)
  - tests/test_outputs.py: pytest test class that validates the solution

Scoring: run pytest against the agent's workspace after task completion.
The evaluate hook in the container runs pytest and reports pass/fail.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import toml

from simple_agent_lab.evals.protocols import LaunchSpec


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Walk data/tasks/{task_slug}/ and load task.toml + instruction.md.

    Returns a flat list of task dicts with all metadata needed by the suite.
    """
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    tasks: list[dict[str, Any]] = []
    for task_toml in sorted(tasks_dir.glob("*/task.toml")):
        task_dir = task_toml.parent
        slug = task_dir.name

        try:
            meta = toml.loads(task_toml.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Read instruction.md
        instruction_path = task_dir / "instruction.md"
        instruction = ""
        if instruction_path.exists():
            instruction = instruction_path.read_text(encoding="utf-8")

        # Read test code
        test_code = ""
        tests_dir = task_dir / "tests"
        if tests_dir.exists():
            parts = []
            for py_file in sorted(tests_dir.glob("test_*.py")):
                parts.append(py_file.read_text(encoding="utf-8"))
            test_code = "\n\n".join(parts)

        # Collect environment files (relative paths)
        env_files: list[str] = []
        env_dir = task_dir / "environment"
        if env_dir.exists():
            for f in env_dir.rglob("*"):
                if f.is_file() and f.name != "Dockerfile":
                    env_files.append(str(f.relative_to(task_dir)))

        metadata = meta.get("metadata", {})
        verifier = meta.get("verifier", {})
        agent = meta.get("agent", {})
        env_cfg = meta.get("environment", {})

        # Read env file contents for workspace seeding
        env_files_data: dict[str, str | bytes] = {}
        for rel_path in env_files:
            abs_path = task_dir / rel_path
            if abs_path.exists():
                if rel_path.endswith((".stl", ".bin", ".pkl", ".npy", ".dat")):
                    env_files_data[rel_path] = abs_path.read_bytes()
                else:
                    try:
                        env_files_data[rel_path] = abs_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        env_files_data[rel_path] = abs_path.read_bytes()

        tasks.append({
            "instance_id": slug,
            "difficulty": metadata.get("difficulty", "medium"),
            "category": metadata.get("category", "general"),
            "tags": metadata.get("tags", []),
            "verifier_timeout": verifier.get("timeout_sec", 900),
            "agent_timeout": agent.get("timeout_sec", 900),
            "env_cpus": env_cfg.get("cpus", 1),
            "env_memory_mb": env_cfg.get("memory_mb", 4096),
            "instruction": instruction,
            "test_code": test_code,
            "env_files": env_files,
        })

    return tasks


class SkillsBenchSuite:
    """Suite for SkillsBench: 88 coding tasks with pytest scoring."""

    name = "skillsbench"
    container_module = "simple_agent_lab.evals.suites.skillsbench.container"

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
        """Agent-visible input: instruction prompt + environment info."""
        return {
            "instance_id": instance["instance_id"],
            "instruction": instance["instruction"],
            "env_files": instance.get("env_files", []),
            "timeout": instance.get("agent_timeout", 900),
            "data_dir": str(self._data_dir),
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Gold scoring data: test code for the evaluate hook."""
        return {
            "instance_id": instance["instance_id"],
            "test_code": instance.get("test_code", ""),
            "verifier_timeout": instance.get("verifier_timeout", 900),
        }
