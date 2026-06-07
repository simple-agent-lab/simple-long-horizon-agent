"""ClawBench-Official: 303 real-world agent tasks with pytest scoring.

Each task lives in a directory under data/tasks/{domain}/{task_id}/ containing:
  - task.toml: metadata (id, title, domain, level, timeout, capabilities, etc.)
  - instruction.md: the agent-visible prompt
  - environment/: setup.sh + optional data/ directory (seeded into workspace)
  - verifier/test_output.py: pytest-based verification tests

Scoring: run pytest against the agent's workspace after task completion.
The evaluate hook in the container runs pytest and reports pass/fail per test.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import toml

from simple_agent_lab.evals.protocols import LaunchSpec


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Walk data/tasks/{domain}/{task_id}/ and load each task.toml + instruction.md.

    Returns a flat list of task dicts with all metadata needed by the suite.
    """
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    tasks: list[dict[str, Any]] = []
    for task_toml in sorted(tasks_dir.rglob("task.toml")):
        task_dir = task_toml.parent

        # Parse task.toml
        # Some files have [task] section header, others are top-level
        try:
            parsed = toml.loads(task_toml.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Handle both [task] section header and flat top-level formats
        if "task" in parsed and isinstance(parsed["task"], dict):
            meta = parsed["task"]
        else:
            meta = parsed

        # Read instruction.md
        instruction_path = task_dir / "instruction.md"
        instruction = ""
        if instruction_path.exists():
            instruction = instruction_path.read_text(encoding="utf-8")

        # Read verifier test code (for eval_inputs so container can run pytest)
        verifier_code = ""
        verifier_dir = task_dir / "verifier"
        if verifier_dir.exists():
            parts = []
            for py_file in sorted(verifier_dir.glob("test_*.py")):
                parts.append(py_file.read_text(encoding="utf-8"))
            verifier_code = "\n\n".join(parts)

        # Collect environment data files (paths relative to task dir)
        env_data_files: list[str] = []
        env_data_dir = task_dir / "environment" / "data"
        if env_data_dir.exists():
            for f in env_data_dir.rglob("*"):
                if f.is_file():
                    env_data_files.append(str(f.relative_to(task_dir)))

        # Collect setup.sh path
        setup_script = ""
        setup_path = task_dir / "environment" / "setup.sh"
        if setup_path.exists():
            setup_script = setup_path.read_text(encoding="utf-8")

        # Use the directory structure to derive domain from parent dir name
        relative = task_dir.relative_to(tasks_dir)
        parts = relative.parts
        domain_from_path = parts[0] if len(parts) >= 2 else meta.get("domain", "unknown")

        tasks.append({
            "instance_id": meta.get("id", task_dir.name),
            "title": meta.get("title", ""),
            "domain": meta.get("domain", domain_from_path),
            "level": meta.get("level", ""),
            "track": meta.get("track", "foundation"),
            "description": meta.get("description", ""),
            "timeout": meta.get("timeout", 120),
            "capabilities": meta.get("capabilities", []),
            "tags": meta.get("tags", []),
            "skills_allowed": meta.get("skills_allowed", False),
            "instruction": instruction,
            "verifier_code": verifier_code,
            "env_data_files": env_data_files,
            "setup_script": setup_script,
            "task_dir_relative": str(task_dir.relative_to(data_dir)),
        })

    return tasks


class ClawBenchOfficialSuite:
    """Suite for ClawBench-Official: 303 tasks with pytest scoring."""

    name = "clawbench-official"
    container_module = "simple_agent_lab.evals.suites.clawbench_official.container"

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
        return LaunchSpec(
            image=self._image,
            workdir="/workspace",
        )

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Agent-visible input: instruction prompt + environment data info."""
        return {
            "instance_id": instance["instance_id"],
            "instruction": instance["instruction"],
            "env_data_files": instance.get("env_data_files", []),
            "setup_script": instance.get("setup_script", ""),
            "timeout": instance.get("timeout", 120),
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Gold scoring data: verifier pytest code for the evaluate hook."""
        return {
            "instance_id": instance["instance_id"],
            "verifier_code": instance.get("verifier_code", ""),
        }
