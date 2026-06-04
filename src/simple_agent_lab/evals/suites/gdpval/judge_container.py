"""GDPVal judge container half.

The judge is a separate suite run. It receives the candidate workspace archive,
gold deliverables, references, and rubrics, then writes a strict JSON judgment
file that ``extract_result`` parses into weighted scores.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent

from .assets import write_reference_files
from .judge_mcp import gdpval_judge_agent_context
from .judge_prompts import GDPVAL_JUDGE_SYSTEM_PROMPT
from .judge_scoring import normalize_rubrics, parse_judge_payload, score_judgment
from .tools import make_gdpval_tools

JUDGE_RESULT_FILE = "_gdpval_judge_result.json"


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the GDPVal judge agent with access to candidate/gold inputs."""

    workdir = Path(cwd)
    input_dir = _input_dir_for(workdir)
    return make_llm_agent(
        name="gdpval_judge",
        provider=provider,
        role="Judge GDPVal deliverables and write a JSON verdict.",
        tools=make_gdpval_tools(workdir=workdir, reference_dir=input_dir),
        system_prompt=GDPVAL_JUDGE_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
    )


def agent_context(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    instance: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
):
    """Build a run-scoped judge agent, keeping MCP tools alive during the run."""

    return gdpval_judge_agent_context(
        provider=provider,
        cwd=Path(cwd),
        request_extra=request_extra,
        instance=instance,
        context=context,
        name="gdpval_judge",
        role="Judge GDPVal deliverables and write a JSON verdict.",
        system_prompt=GDPVAL_JUDGE_SYSTEM_PROMPT,
    )


def prepare(workspace: Path, instance: Mapping[str, Any]) -> dict[str, Any]:
    """Stage candidate, gold, and reference files for judge inspection."""

    workdir = Path(workspace)
    workdir.mkdir(parents=True, exist_ok=True)
    input_dir = _input_dir_for(workdir)
    candidate_dir = input_dir / "candidate"
    gold_dir = input_dir / "gold"
    reference_dir = input_dir / "reference"
    for path in (candidate_dir, gold_dir, reference_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidate_manifest = _write_candidate_inputs(instance, candidate_dir)
    gold_manifest = write_reference_files(
        instance.get("gold_file_blobs") or [], gold_dir
    )
    reference_manifest = write_reference_files(
        instance.get("reference_file_blobs") or [], reference_dir
    )
    zip_manifest = _extract_zip_archives(input_dir)

    return {
        "input_dir": str(input_dir),
        "candidate_dir": str(candidate_dir),
        "gold_dir": str(gold_dir),
        "reference_dir": str(reference_dir),
        "candidate_manifest": candidate_manifest,
        "gold_manifest": gold_manifest,
        "reference_manifest": reference_manifest,
        "zip_manifest": zip_manifest,
    }


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    workdir_path = Path(workdir)
    input_dir = _input_dir_for(workdir_path)
    rubrics = normalize_rubrics(instance.get("rubrics"))
    result_path = workdir_path / JUDGE_RESULT_FILE
    candidate_result = instance.get("candidate_result") or {}
    return "\n".join(
        [
            "Judge this GDPVal candidate submission.",
            "",
            "## Paths",
            f"- WORKDIR: {workdir_path}",
            f"- CANDIDATE_DIR: {input_dir / 'candidate'}",
            f"- GOLD_DIR: {input_dir / 'gold'}",
            f"- REFERENCE_DIR: {input_dir / 'reference'}",
            f"- ZIP_EXTRACTS: {input_dir / '__zip_extracts'}",
            f"- REQUIRED_OUTPUT_JSON: {result_path}",
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
            "- Inspect the candidate files and gold/reference files as needed.",
            "- Write the judgment JSON to REQUIRED_OUTPUT_JSON.",
            "- Include exactly one rubric_results item for each rubric index above.",
            "- Do not write the judgment anywhere else.",
        ]
    )


def apply_oracle(workspace: Path, instance: Mapping[str, Any]) -> None:
    """Model-free smoke path: write a perfect score for every staged rubric."""

    rubrics = normalize_rubrics(instance.get("rubrics"))
    payload = {
        "rubric_results": [
            {
                "index": item["index"],
                "criterion": item["criterion"],
                "grade": 1.0,
                "explanation": "oracle judge marks the rubric satisfied",
            }
            for item in rubrics
        ],
        "overall_explanation": "oracle judge result",
    }
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
    """Parse the judge JSON file and compute weighted rubric scores."""

    workdir = Path(workspace)
    result_file = workdir / JUDGE_RESULT_FILE
    base: dict[str, Any] = {
        "task_id": str(instance.get("task_id") or instance.get("instance_id") or ""),
        "judge_result_file": str(result_file),
    }
    if context:
        base["input_dir"] = str(context.get("input_dir") or "")
        base["candidate_dir"] = str(context.get("candidate_dir") or "")
        base["gold_dir"] = str(context.get("gold_dir") or "")
        base["reference_dir"] = str(context.get("reference_dir") or "")

    if not result_file.is_file():
        return {
            **base,
            "status": "judge_result_missing",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": sum(
                item["weight"] for item in normalize_rubrics(instance.get("rubrics"))
            ),
            "rubric_results": [],
            "overall_explanation": "judge did not write the required JSON file",
        }
    raw = result_file.read_text(encoding="utf-8", errors="replace")
    try:
        payload = parse_judge_payload(raw)
        scored = score_judgment(payload, instance.get("rubrics"))
    except ValueError as exc:
        return {
            **base,
            "status": "judge_result_invalid",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": sum(
                item["weight"] for item in normalize_rubrics(instance.get("rubrics"))
            ),
            "rubric_results": [],
            "overall_explanation": f"{type(exc).__name__}: {exc}",
            "raw_judge_result": raw[:20_000],
        }
    return {
        **base,
        **scored,
        "raw_judge_result_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _input_dir_for(workdir: Path) -> Path:
    return Path(workdir).parent / "judge_inputs"


def _candidate_summary(candidate_result: Any) -> dict[str, Any]:
    if not isinstance(candidate_result, Mapping):
        return {}
    files = candidate_result.get("files") or []
    if not isinstance(files, list):
        files = []
    return {
        "status": candidate_result.get("status"),
        "file_count": len(files),
        "files": [
            {
                "relative_path": item.get("relative_path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
            }
            for item in files[:50]
            if isinstance(item, Mapping)
        ],
    }


def _write_candidate_inputs(
    instance: Mapping[str, Any], candidate_dir: Path
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    archive_b64 = instance.get("candidate_workspace_archive_base64")
    if isinstance(archive_b64, str) and archive_b64:
        data = base64.b64decode(archive_b64)
        archive_manifest = _extract_tar_gz(data, candidate_dir)
        manifest.append(
            {
                "name": "candidate_workspace_archive",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "entries": archive_manifest,
            }
        )

    file_blobs = instance.get("candidate_file_blobs") or []
    if file_blobs:
        manifest.extend(write_reference_files(file_blobs, candidate_dir))

    (candidate_dir / "_sal_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _extract_tar_gz(data: bytes, dest: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if _unsafe_archive_name(member.name):
                entries.append({"name": member.name, "status": "skipped_unsafe"})
                continue
            target = (dest / member.name).resolve()
            if not _inside(target, dest.resolve()):
                entries.append({"name": member.name, "status": "skipped_unsafe"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            target.write_bytes(payload)
            entries.append(
                {
                    "name": member.name,
                    "path": str(target),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return entries


def _extract_zip_archives(input_dir: Path) -> list[dict[str, Any]]:
    extract_root = input_dir / "__zip_extracts"
    manifest: list[dict[str, Any]] = []
    for archive in sorted(input_dir.rglob("*.zip")):
        if extract_root in archive.parents:
            continue
        rel = archive.relative_to(input_dir)
        dest = extract_root / rel.with_suffix("")
        item: dict[str, Any] = {
            "archive": str(archive),
            "extract_dir": str(dest),
            "entries": [],
            "status": "ok",
        }
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if _unsafe_archive_name(info.filename):
                        item["entries"].append(
                            {"name": info.filename, "status": "skipped_unsafe"}
                        )
                        continue
                    target = (dest / info.filename).resolve()
                    if not _inside(target, dest.resolve()):
                        item["entries"].append(
                            {"name": info.filename, "status": "skipped_unsafe"}
                        )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    payload = zf.read(info)
                    target.write_bytes(payload)
                    item["entries"].append(
                        {
                            "name": info.filename,
                            "path": str(target),
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
        except Exception as exc:
            item["status"] = "error"
            item["error"] = f"{type(exc).__name__}: {exc}"
        manifest.append(item)
    if manifest:
        extract_root.mkdir(parents=True, exist_ok=True)
        (extract_root / "_sal_zip_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def _unsafe_archive_name(name: str) -> bool:
    normalized = Path(name.replace("\\", "/"))
    return (
        not name
        or normalized.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.parts)
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
