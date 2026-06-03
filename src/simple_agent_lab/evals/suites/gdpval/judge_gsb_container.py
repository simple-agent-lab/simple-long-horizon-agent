"""GDPVal GSB judge container half.

The GSB judge is a second-stage run. It receives candidate deliverables, gold
deliverables, references, and rubrics, then writes a strict JSON judgment with
forward and reverse A/B comparisons.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent

from .judge_container import _candidate_summary, _input_dir_for, prepare as prepare
from .judge_gsb_prompts import GDPVAL_GSB_JUDGE_SYSTEM_PROMPT
from .judge_scoring import (
    normalize_rubrics,
    parse_gsb_judge_payload,
    score_gsb_judgment,
)
from .tools import make_gdpval_tools

JUDGE_RESULT_FILE = "_gdpval_gsb_judge_result.json"


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the GDPVal GSB judge agent with access to candidate/gold inputs."""

    workdir = Path(cwd)
    input_dir = _input_dir_for(workdir)
    return make_llm_agent(
        name="gdpval_gsb_judge",
        provider=provider,
        role="Compare GDPVal deliverables and write a GSB JSON verdict.",
        tools=make_gdpval_tools(workdir=workdir, reference_dir=input_dir),
        system_prompt=GDPVAL_GSB_JUDGE_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    workdir_path = Path(workdir)
    input_dir = _input_dir_for(workdir_path)
    rubrics = normalize_rubrics(instance.get("rubrics"))
    result_path = workdir_path / JUDGE_RESULT_FILE
    candidate_result = instance.get("candidate_result") or {}
    return "\n".join(
        [
            "Judge this GDPVal candidate submission with GSB comparison.",
            "",
            "## Paths",
            f"- WORKDIR: {workdir_path}",
            f"- CANDIDATE_DIR: {input_dir / 'candidate'}",
            f"- GOLD_DIR: {input_dir / 'gold'}",
            f"- REFERENCE_DIR: {input_dir / 'reference'}",
            f"- ZIP_EXTRACTS: {input_dir / '__zip_extracts'}",
            f"- REQUIRED_OUTPUT_JSON: {result_path}",
            "",
            "## Direction Definitions",
            "- reverse: A is GOLD_DIR, B is CANDIDATE_DIR.",
            "- forward: A is CANDIDATE_DIR, B is GOLD_DIR.",
            "",
            "## Original Task Prompt",
            str(instance.get("prompt") or ""),
            "",
            "## Candidate Result Summary",
            json.dumps(
                _candidate_summary(candidate_result), ensure_ascii=False, indent=2
            ),
            "",
            "## Rubrics",
            json.dumps(rubrics, ensure_ascii=False, indent=2),
            "",
            "## Instructions",
            "- Inspect candidate, gold, and reference files as needed.",
            "- Write the GSB judgment JSON to REQUIRED_OUTPUT_JSON.",
            "- Include exactly one rubrics_result item for each rubric index in "
            "both reverse and forward.",
            "- Use the exact GSB labels specified in the system prompt.",
            "- Do not write the judgment anywhere else.",
        ]
    )


def apply_oracle(workspace: Path, instance: Mapping[str, Any]) -> None:
    """Model-free smoke path: write a deterministic tie against gold."""

    rubrics = normalize_rubrics(instance.get("rubrics"))
    reverse = _oracle_direction(
        rubrics,
        grade_a=1.0,
        grade_b=1.0,
        gsb="A=B",
        final_gsb="A=B",
        explanation="oracle judge marks candidate and gold as tied",
    )
    payload = {"reverse": reverse, "forward": reverse}
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / JUDGE_RESULT_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse the GSB judge JSON file and compute GDPVal GSB scores."""

    workdir = Path(workspace)
    result_file = workdir / JUDGE_RESULT_FILE
    max_score = sum(
        item["weight"] for item in normalize_rubrics(instance.get("rubrics"))
    )
    base: dict[str, Any] = {
        "task_id": str(instance.get("task_id") or instance.get("instance_id") or ""),
        "judge_mode": "gsb",
        "judge_result_file": str(result_file),
    }
    if context:
        base["input_dir"] = str(context.get("input_dir") or "")
        base["candidate_dir"] = str(context.get("candidate_dir") or "")
        base["gold_dir"] = str(context.get("gold_dir") or "")
        base["reference_dir"] = str(context.get("reference_dir") or "")
        base["gold_manifest"] = context.get("gold_manifest") or []
        if not _has_available_files(context.get("gold_manifest")):
            return {
                **base,
                "status": "gold_deliverables_missing",
                "score": 0.0,
                "earned_score": 0.0,
                "max_score": max_score,
                "overall_explanation_reverse": (
                    "no readable standard-answer deliverable files were staged"
                ),
                "overall_explanation_forward": (
                    "no readable standard-answer deliverable files were staged"
                ),
            }

    if not result_file.is_file():
        return {
            **base,
            "status": "judge_result_missing",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": max_score,
            "overall_explanation_reverse": (
                "judge did not write the required GSB JSON file"
            ),
            "overall_explanation_forward": (
                "judge did not write the required GSB JSON file"
            ),
        }
    raw = result_file.read_text(encoding="utf-8", errors="replace")
    try:
        payload = parse_gsb_judge_payload(raw)
        scored = score_gsb_judgment(payload, instance.get("rubrics"))
    except ValueError as exc:
        return {
            **base,
            "status": "judge_result_invalid",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": max_score,
            "overall_explanation_reverse": f"{type(exc).__name__}: {exc}",
            "overall_explanation_forward": f"{type(exc).__name__}: {exc}",
            "raw_judge_result": raw[:20_000],
        }
    return {
        **base,
        **scored,
        "raw_judge_result_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _oracle_direction(
    rubrics: list[dict[str, Any]],
    *,
    grade_a: float,
    grade_b: float,
    gsb: str,
    final_gsb: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "rubrics_result": [
            {
                "index": item["index"],
                "score": item["weight"],
                "criterion": item["criterion"],
                "grade_A": grade_a,
                "grade_B": grade_b,
                "gsb": gsb,
                "grade_explanation": explanation,
            }
            for item in rubrics
        ],
        "overall": {
            "overall_explanation": explanation,
            "final_gsb": final_gsb,
        },
    }


def _has_available_files(manifest: Any) -> bool:
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        return False
    for item in manifest:
        if not isinstance(item, Mapping):
            continue
        if not item.get("missing") and item.get("path"):
            return True
    return False
