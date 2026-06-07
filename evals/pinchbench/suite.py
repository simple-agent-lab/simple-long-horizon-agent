"""PinchBench: 23 tasks with embedded grade() function scoring.

Task format: markdown files with YAML frontmatter + Prompt + Automated Checks sections.
Scoring: grade(transcript, workspace_path) function embedded in task markdown.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from simple_agent_lab.evals.protocols import LaunchSpec


def _extract_section(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_tasks(data_dir: Path) -> list[dict[str, Any]]:
    """Parse tasks/*.md → list of task dicts."""
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    tasks = []
    for md in sorted(tasks_dir.glob("*.md")):
        if md.name.startswith("TASK_TEMPLATE") or md.name.startswith("_"):
            continue
        content = md.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter: dict[str, Any] = {}
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1)) or {}
            except Exception:
                pass

        # Extract ## Prompt section
        prompt = ""
        prompt_match = re.search(r"## Prompt\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
        if prompt_match:
            prompt = prompt_match.group(1).strip()

        # Extract ## Automated Checks python code
        grade_code = ""
        checks_match = re.search(
            r"## Automated Checks.*?```python\s*\n(.*?)```", content, re.DOTALL
        )
        if checks_match:
            grade_code = checks_match.group(1).strip()

        # Extract other sections
        expected_behavior = _extract_section(
            r"## Expected Behavior\s*\n(.*?)(?=\n## |\Z)", content
        )
        grading_criteria = _extract_section(
            r"## Grading Criteria\s*\n(.*?)(?=\n## |\Z)", content
        )
        llm_judge_rubric = _extract_section(
            r"## LLM Judge Rubric\s*\n(.*?)(?=\n## |\Z)", content
        )

        tid = frontmatter.get("id", md.stem)
        timeout = int(frontmatter.get("timeout_seconds", 120))
        workspace_files = frontmatter.get("workspace_files", [])
        grading_type = frontmatter.get("grading_type", "automated")

        tasks.append({
            "instance_id": tid,
            "prompt": prompt,
            "grade_code": grade_code,
            "timeout": timeout,
            "workspace_files": workspace_files,
            "grading_type": grading_type,
            "llm_judge_rubric": llm_judge_rubric,
            "expected_behavior": expected_behavior,
            "grading_criteria": grading_criteria,
        })

    return tasks


class PinchBenchSuite:
    """Suite for PinchBench: 23 tasks with automated grading."""

    name = "pinchbench"
    container_module = "simple_agent_lab.evals.suites.pinchbench.container"

    def __init__(self, *, data_dir: str | Path | None = None, image: str = "clawbase-sal:v1") -> None:
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
        return {
            "instance_id": instance["instance_id"],
            "prompt": instance["prompt"],
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        grade_code = instance.get("grade_code", "")
        if not grade_code:
            return None
        return {
            "instance_id": instance["instance_id"],
            "grade_code": grade_code,
            "grading_type": instance.get("grading_type", "automated"),
        }
