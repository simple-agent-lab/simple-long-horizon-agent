"""AgentBench container half: build task + extract result + evaluate (Layer 0).

The agent executes the user_message in the workspace. The evaluate hook
performs Layer 0 structural checks (file-exists, content-contains,
word-count-range). Layers 1-3 are scored host-side.

Overall logic:
1. build_task: seeds input files, returns the user_message as the agent prompt
2. extract_result: lists workspace files and their sizes
3. evaluate: runs Layer 0 validators against workspace outputs
"""
from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="agentbench_agent",
        role="Expert agent handling real-world tasks across multiple domains.",
        system_prompt=(
            "You are an expert agent. Complete the given task by reading any "
            "input files, performing the required analysis or actions, and "
            "creating the requested output files in the workspace directory. "
            "Be thorough, accurate, and follow all instructions precisely."
        ),
        flavor="bash_task",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the agent-visible task from the instance data.

    Seeds input files from the data directory into the workspace and returns
    the user_message as the task prompt.
    """
    workspace = Path(workdir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy input files from data directory to workspace
    data_dir = instance.get("data_dir", "")
    task_dir = instance.get("task_dir", "")
    input_file_names = instance.get("input_file_names", [])

    if data_dir and task_dir and input_file_names:
        src_dir = Path(data_dir) / task_dir / "inputs"
        if src_dir.exists():
            for fname in input_file_names:
                src = src_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(workspace / fname))

    return str(instance["user_message"])


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """List workspace files with sizes for downstream scoring."""
    files = []
    for f in sorted(workspace.rglob("*")):
        if f.is_file() and not f.name.startswith("_"):
            files.append({
                "path": str(f.relative_to(workspace)),
                "size": f.stat().st_size,
            })

    file_paths = [f["path"] for f in files]

    return {
        "instance_id": instance.get("instance_id", ""),
        "workspace_files": file_paths,
        "file_details": files,
        "status": "completed",
    }


def evaluate(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Layer 0 structural checks against the workspace.

    Layer 0 validates:
    - file-exists: expected output files are present
    - content-contains: files contain required section keywords
    - word-count-range: files have reasonable word counts

    Returns per-validator scores and an aggregate L0 score (0-100).
    """
    eval_data = (context or {}).get("eval", {})
    instance_id = eval_data.get("instance_id", instance.get("instance_id", ""))
    expected_outputs = eval_data.get("expected_outputs", [])

    if not expected_outputs:
        return {
            "instance_id": instance_id,
            "layer0_score": 0,
            "validators": [],
            "error": "no expected_outputs",
        }

    validator_results = []
    total_points = 0
    earned_points = 0

    for output_spec in expected_outputs:
        pattern = output_spec.get("pattern", "")
        required = output_spec.get("required", True)
        validators = output_spec.get("validators", [])

        for validator in validators:
            v_type = validator.get("type", "")
            v_points = 30  # default points per validator
            v_earned = 0

            if v_type == "file-exists":
                target = workspace / pattern
                exists = target.exists()
                v_earned = v_points if exists else 0
                validator_results.append({
                    "type": "file-exists",
                    "pattern": pattern,
                    "passed": exists,
                    "points": v_earned,
                    "max_points": v_points,
                })

            elif v_type == "content-contains":
                target = workspace / pattern
                sections = validator.get("sections", [])
                if target.exists():
                    content = target.read_text(encoding="utf-8", errors="replace").lower()
                    matched = [s for s in sections if s.lower() in content]
                    if sections:
                        ratio = len(matched) / len(sections)
                    else:
                        ratio = 0
                    v_earned = round(v_points * ratio)
                else:
                    matched = []
                    ratio = 0

                validator_results.append({
                    "type": "content-contains",
                    "pattern": pattern,
                    "sections_matched": matched,
                    "sections_total": len(sections),
                    "ratio": round(ratio, 2),
                    "passed": ratio >= 0.5,
                    "points": v_earned,
                    "max_points": v_points,
                })

            elif v_type == "word-count-range":
                target = workspace / pattern
                min_words = validator.get("min", 0)
                max_words = validator.get("max", float("inf"))

                if target.exists():
                    word_count = len(target.read_text(encoding="utf-8", errors="replace").split())
                    in_range = min_words <= word_count <= max_words
                    # Within 2x range gets partial credit
                    near_range = (
                        (min_words / 2 <= word_count <= max_words * 2)
                        if min_words > 0
                        else (word_count <= max_words * 2)
                    )
                    if in_range:
                        v_earned = v_points
                    elif near_range:
                        v_earned = v_points // 2
                    else:
                        v_earned = 0
                else:
                    word_count = 0
                    in_range = False
                    near_range = False

                validator_results.append({
                    "type": "word-count-range",
                    "pattern": pattern,
                    "word_count": word_count,
                    "range": [min_words, max_words],
                    "in_range": in_range,
                    "passed": in_range,
                    "points": v_earned,
                    "max_points": v_points,
                })

            else:
                # Unknown validator type — skip
                continue

            total_points += v_points
            earned_points += v_earned

    layer0_score = round((earned_points / total_points) * 100, 1) if total_points > 0 else 0

    return {
        "instance_id": instance_id,
        "layer0_score": layer0_score,
        "validators": validator_results,
        "total_points": total_points,
        "earned_points": earned_points,
    }
