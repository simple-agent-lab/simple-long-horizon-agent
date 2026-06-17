"""Host-side GDPVal judge suite and instance builder."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec, RunArtifacts
from simple_agent_lab.evals.suites.gdpval.assets import normalize_reference_file_inputs
from simple_agent_lab.evals.suites.gdpval.judge_mcp import (
    DEFAULT_JUDGE_TOOL_MODE,
    JudgeToolMode,
    normalize_judge_tool_mode,
)

from .suite import DEFAULT_GDPVAL_IMAGE, DEFAULT_WORKDIR_PREFIX


class GdpvalJudgeSuite:
    """A follow-up GDPVal rubric judge.

    The solver run remains gold-free. This suite is run as a second stage over
    the solver artifacts and receives rubrics/gold deliverables in its own
    trusted instance.
    """

    name = "gdpval_judge"
    container_module = "simple_agent_lab.evals.suites.gdpval.judge_container"

    def __init__(
        self,
        *,
        image: str = DEFAULT_GDPVAL_IMAGE,
        workdir_prefix: str = DEFAULT_WORKDIR_PREFIX,
        reference_root: str | Path | None = None,
        deliverable_root: str | Path | None = None,
        network_mode: str | None = "host",
        platform: str | None = None,
        judge_tool_mode: str = DEFAULT_JUDGE_TOOL_MODE,
    ) -> None:
        self.image = image
        self.workdir_prefix = workdir_prefix.rstrip("/")
        self.reference_root = Path(reference_root).resolve() if reference_root else None
        self.deliverable_root = (
            Path(deliverable_root).resolve() if deliverable_root else None
        )
        self.network_mode = network_mode
        self.platform = platform
        self.judge_tool_mode: JudgeToolMode = normalize_judge_tool_mode(judge_tool_mode)

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        task_id = _task_id(instance)
        return LaunchSpec(
            image=self.image,
            workdir=f"{self.workdir_prefix}/{_safe_path_part(task_id)}/judge_workdir",
            shell=("bash", "-lc"),
            entrypoint="",
            platform=self.platform,
            network_mode=self.network_mode,
        )

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None

    def build_instance(
        self,
        source_instance: Mapping[str, Any],
        *,
        candidate_result: Mapping[str, Any],
        candidate_artifacts: RunArtifacts,
    ) -> dict[str, Any]:
        task_id = _task_id(source_instance)
        archive_base64 = _candidate_archive_base64(
            candidate_result, candidate_artifacts
        )
        gold_blobs = normalize_reference_file_inputs(
            _gold_field(
                source_instance,
                prefer_urls=self.deliverable_root is None,
            ),
            reference_root=self.deliverable_root,
            task_id=task_id,
        )
        reference_blobs = normalize_reference_file_inputs(
            _reference_field(
                source_instance,
                prefer_urls=self.reference_root is None,
            ),
            reference_root=self.reference_root,
            task_id=task_id,
        )
        return {
            "instance_id": task_id,
            "task_id": task_id,
            "prompt": str(
                source_instance.get("prompt")
                or source_instance.get("prompt_en")
                or source_instance.get("question")
                or ""
            ),
            "rubrics": _rubrics_field(source_instance),
            "candidate_result": dict(candidate_result),
            "candidate_run_dir": str(candidate_artifacts.run_dir),
            "candidate_workspace_archive_base64": archive_base64,
            "candidate_file_blobs": (
                [] if archive_base64 else _candidate_file_blobs(candidate_result)
            ),
            "judge_tool_mode": self.judge_tool_mode,
            "gold_files": _public_entries(gold_blobs),
            "gold_file_blobs": gold_blobs,
            "reference_file_blobs": reference_blobs,
        }


def _candidate_archive_base64(
    candidate_result: Mapping[str, Any], artifacts: RunArtifacts
) -> str:
    artifact_key = str(candidate_result.get("workspace_archive_artifact") or "")
    candidates: list[Path] = []
    if artifact_key:
        candidates.append(artifacts.run_dir / artifact_key)
    archive_path = candidate_result.get("workspace_archive_path")
    if isinstance(archive_path, str) and archive_path:
        candidates.append(Path(archive_path))
    for path in candidates:
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    return ""


def _candidate_file_blobs(candidate_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    blobs: list[dict[str, Any]] = []
    files = candidate_result.get("files") or []
    if not isinstance(files, list):
        return blobs
    for item in files:
        if not isinstance(item, Mapping):
            continue
        path = Path(str(item.get("path") or ""))
        if not path.is_file():
            continue
        data = path.read_bytes()
        name = str(item.get("relative_path") or path.name)
        blobs.append(
            {
                "name": name,
                "source": str(path),
                "size_bytes": len(data),
                "sha256": item.get("sha256"),
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    return blobs


def _rubrics_field(instance: Mapping[str, Any]) -> Any:
    for key in ("rubrics", "rubric_json"):
        if key in instance:
            return instance[key]
    return []


def _gold_field(instance: Mapping[str, Any], *, prefer_urls: bool = False) -> Any:
    if prefer_urls and instance.get("deliverable_file_urls"):
        urls = _as_list(instance.get("deliverable_file_urls"))
        names = _as_list(instance.get("deliverable_files"))
        return [
            {
                "name": names[index] if index < len(names) else url,
                "path": names[index] if index < len(names) else "",
                "url": url,
            }
            for index, url in enumerate(urls)
        ]
    for key in ("deliverable_files", "deliverable_file_urls"):
        if key in instance:
            return instance[key]
    return []


def _reference_field(instance: Mapping[str, Any], *, prefer_urls: bool = False) -> Any:
    if prefer_urls and instance.get("reference_file_urls"):
        urls = _as_list(instance.get("reference_file_urls"))
        names = _as_list(instance.get("reference_files"))
        return [
            {
                "name": names[index] if index < len(names) else url,
                "path": names[index] if index < len(names) else "",
                "url": url,
            }
            for index, url in enumerate(urls)
        ]
    for key in ("reference_files", "reference_file_urls", "reference_files_json"):
        if key in instance:
            return instance[key]
    return []


def _public_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for entry in entries:
        item = {
            "name": entry.get("name") or entry.get("source") or "file",
            "source": entry.get("source") or "",
            "missing": bool(entry.get("missing")),
        }
        if "size_bytes" in entry:
            item["size_bytes"] = entry["size_bytes"]
        if "sha256" in entry:
            item["sha256"] = entry["sha256"]
        public.append(item)
    return public


def _task_id(instance: Mapping[str, Any]) -> str:
    return str(
        instance.get("instance_id")
        or instance.get("task_id")
        or instance.get("id")
        or instance.get("taskId")
        or ""
    ).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return _as_list(parsed)
        return [text]
    if isinstance(value, Mapping):
        return [str(item) for item in value.values() if str(item).strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _safe_path_part(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    return safe.strip("._") or "task"


class GdpvalGsbJudgeSuite(GdpvalJudgeSuite):
    """A GDPVal GSB judge comparing candidate deliverables to gold files."""

    name = "gdpval_gsb_judge"
    container_module = "simple_agent_lab.evals.suites.gdpval.judge_gsb_container"
