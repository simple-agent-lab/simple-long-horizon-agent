"""GDPVal as a Simple Agent Lab `Suite`.

This is the host half. It launches the GDPVal sandbox image, strips gold fields
from the agent-visible input, and stages reference files as private blobs for
the container half to write under REFERENCE_DIR.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec
from simple_agent_lab.evals.suites.gdpval.assets import normalize_reference_file_inputs

DEFAULT_GDPVAL_IMAGE = "hub.byted.org/apihub/gdpeval:1.0.0"
DEFAULT_WORKDIR_PREFIX = "/app/workspace/gdpevals"

_GOLD_KEYS = {
    "answer",
    "bonus_rubrics",
    "deduction_rubrics",
    "deliverable_file_urls",
    "deliverable_files",
    "gold",
    "judge",
    "judge_config",
    "reference_answer",
    "rubric_json",
    "rubrics",
    "standard_answer",
}


class GdpvalSuite:
    """Solver-only GDPVal suite.

    First-version scope: run the context-managed tool-call solver and collect
    workspace artifacts. Judge scoring is intentionally left out.
    """

    name = "gdpval"
    container_module = "simple_agent_lab.evals.suites.gdpval.container"

    def __init__(
        self,
        *,
        image: str = DEFAULT_GDPVAL_IMAGE,
        workdir_prefix: str = DEFAULT_WORKDIR_PREFIX,
        reference_root: str | Path | None = None,
        network_mode: str | None = "host",
        platform: str | None = None,
    ) -> None:
        self.image = image
        self.workdir_prefix = workdir_prefix.rstrip("/")
        self.reference_root = Path(reference_root).resolve() if reference_root else None
        self.network_mode = network_mode
        self.platform = platform

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        task_id = _task_id(instance)
        return LaunchSpec(
            image=self.image,
            workdir=f"{self.workdir_prefix}/{_safe_path_part(task_id)}/workdir",
            shell=("bash", "-lc"),
            entrypoint="",
            platform=self.platform,
            network_mode=self.network_mode,
        )

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        record = {k: v for k, v in dict(instance).items() if k not in _GOLD_KEYS}
        task_id = _task_id(instance)
        prompt = str(
            instance.get("prompt")
            or instance.get("prompt_en")
            or instance.get("question")
            or ""
        )
        reference_blobs = normalize_reference_file_inputs(
            _reference_field(instance, prefer_urls=self.reference_root is None),
            reference_root=self.reference_root,
            task_id=task_id,
        )
        record.update(
            {
                "instance_id": task_id,
                "task_id": task_id,
                "prompt": prompt,
                "reference_files": [
                    _public_reference_entry(x) for x in reference_blobs
                ],
                "reference_file_blobs": reference_blobs,
                "solver_agent_mode": "tool-call-context-managed",
            }
        )
        return record

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Solver-only first version: do not stage judge/gold fields at all."""

        return None


def _task_id(instance: Mapping[str, Any]) -> str:
    return str(
        instance.get("instance_id")
        or instance.get("task_id")
        or instance.get("id")
        or instance.get("taskId")
        or ""
    ).strip()


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


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, Mapping):
        return [str(item) for item in value.values() if str(item).strip()]
    return [str(value)]


def _public_reference_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        "name": entry.get("name") or entry.get("source") or "reference",
        "source": entry.get("source") or "",
        "missing": bool(entry.get("missing")),
    }
    if "size_bytes" in entry:
        public["size_bytes"] = entry["size_bytes"]
    if "sha256" in entry:
        public["sha256"] = entry["sha256"]
    return public


def _safe_path_part(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    return safe.strip("._") or "task"


def dumps_for_debug(value: Mapping[str, Any]) -> str:
    """Small helper for ad hoc inspection in notebooks/scripts."""

    return json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
