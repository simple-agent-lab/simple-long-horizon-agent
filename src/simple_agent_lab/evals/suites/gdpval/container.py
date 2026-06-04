"""GDPVal container half for the Simple Agent Lab eval framework.

First-version scope: run the solver only, with a no-web tool-call agent, and
collect the workspace deliverables. It intentionally does not import or call
swalm.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
from simple_agent_lab.evals.stores import container_store_from_env
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent

from .assets import write_reference_files
from .prompts import GDPVAL_SYSTEM_PROMPT
from .tools import make_gdpval_tools


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the first-version GDPVal solver."""

    workdir = Path(cwd)
    reference_dir = _reference_dir_for(workdir)
    return make_llm_agent(
        name="gdpval_solver",
        provider=provider,
        role="Complete GDPVal tasks by creating deliverables in WORKDIR.",
        tools=make_gdpval_tools(
            workdir=workdir,
            reference_dir=reference_dir,
            profile="bash_fileops",
        ),
        system_prompt=GDPVAL_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
    )


def prepare(workspace: Path, instance: Mapping[str, Any]) -> dict[str, Any]:
    workdir = Path(workspace)
    workdir.mkdir(parents=True, exist_ok=True)
    reference_dir = _reference_dir_for(workdir)
    reference_manifest = write_reference_files(
        instance.get("reference_file_blobs") or [],
        reference_dir,
    )
    return {
        "workdir": str(workdir),
        "reference_dir": str(reference_dir),
        "reference_manifest": reference_manifest,
    }


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    reference_dir = _reference_dir_for(Path(workdir))
    prompt = str(instance.get("prompt") or "")
    task_id = str(instance.get("task_id") or instance.get("instance_id") or "")
    references = instance.get("reference_files") or []
    reference_lines = _reference_lines(references, reference_dir=reference_dir)
    return "\n".join(
        [
            "Complete this GDPVal task.",
            "",
            "## Sandbox Paths",
            f"- TASK_ID: {task_id}",
            f"- WORKDIR: {workdir}",
            f"- REFERENCE_DIR: {reference_dir}",
            "",
            "## Reference Files",
            reference_lines or "- No reference files were staged.",
            "",
            "## Task Prompt",
            prompt,
            "",
            "## Completion Requirements",
            "- Create all requested deliverables under WORKDIR.",
            "- Verify generated files exist and are non-empty when practical.",
            "- Finish with a final assistant message containing "
            "<FINAL_ANSWER>...</FINAL_ANSWER>.",
            "- Treat those tags as plain final-message text, not as a tool call.",
            "- Put generated file paths and the completion summary inside those tags.",
        ]
    )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect generated deliverables and persist a workspace archive if possible."""

    workdir = Path(workspace)
    files = _workspace_manifest(workdir)
    archive_bytes = _archive_workspace(workdir)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    archive_artifact = ""
    stored = False
    if "SAL_STORE" in os.environ:
        try:
            store = container_store_from_env()
            archive_artifact = "out/workspace.tar.gz"
            store.put(archive_artifact, archive_bytes)
            stored = True
        except Exception:
            archive_artifact = ""

    result: dict[str, Any] = {
        "task_id": str(instance.get("task_id") or instance.get("instance_id") or ""),
        "status": "solver_finished",
        "workdir": str(workdir),
        "reference_dir": str(_reference_dir_for(workdir)),
        "files": files,
        "workspace_archive_sha256": archive_sha,
        "workspace_archive_bytes": len(archive_bytes),
    }
    if stored:
        result["workspace_archive_artifact"] = archive_artifact
    else:
        archive_path = workdir / "_sal_workspace.tar.gz"
        archive_path.write_bytes(archive_bytes)
        result["workspace_archive_path"] = str(archive_path)
    return result


def _reference_dir_for(workdir: Path) -> Path:
    return workdir.parent / "reference_task_id_files"


def _reference_lines(references: Any, *, reference_dir: Path) -> str:
    if not isinstance(references, list):
        return ""
    lines: list[str] = []
    for item in references:
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("path") or item.get("source")
            missing = " (missing)" if item.get("missing") else ""
            if item.get("missing"):
                source = item.get("source") or name
                lines.append(f"- {name}{missing}; source: {source}")
            else:
                lines.append(f"- {reference_dir / str(name)}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _workspace_manifest(workdir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not workdir.exists():
        return out
    for path in sorted(p for p in workdir.rglob("*") if p.is_file()):
        if path.name.startswith("_sal_"):
            continue
        data = path.read_bytes()
        out.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(workdir).as_posix(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return out


def _archive_workspace(workdir: Path) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with tarfile.open(temp_path, "w:gz") as tar:
            if workdir.exists():
                for path in sorted(p for p in workdir.rglob("*") if p.is_file()):
                    if path.name.startswith("_sal_"):
                        continue
                    tar.add(
                        path, arcname=str(Path("workdir") / path.relative_to(workdir))
                    )
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)
