"""Run artifacts: dataclass, path-safety helpers, and normalization.

Artifacts are the raw products (diffs, logs, generated files) a run wants kept
verbatim under a memory namespace's ``runs/<run_id>/artifacts/`` directory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable

from simple_agent_lab.memory.base import MemoryContext
from simple_agent_lab.memory.transcript import final_submission_from_state


@dataclass(frozen=True)
class FilesystemArtifact:
    """One durable run artifact to store under ``artifacts/``."""

    name: str
    content: str
    description: str = ""


ArtifactBuilder = Callable[[MemoryContext], Iterable[FilesystemArtifact]]


def default_artifacts(ctx: MemoryContext) -> tuple[FilesystemArtifact, ...]:
    """Build generic artifacts from explicit memory data and final submissions."""

    artifacts = list(_coerce_artifacts(ctx.data.get("memory_artifacts")))
    if ctx.state is not None and not artifacts:
        artifacts = list(_coerce_artifacts(ctx.state.data.get("memory_artifacts")))

    submission = final_submission_from_state(ctx.state) if ctx.state is not None else ""
    names = {artifact.name for artifact in artifacts}
    if submission and "submission.txt" not in names:
        artifacts.append(
            FilesystemArtifact(
                name="submission.txt",
                content=submission,
                description="Final run submission or primary output artifact.",
            )
        )
    return tuple(artifacts)


def normalize_artifacts(
    artifacts: Iterable[FilesystemArtifact],
) -> tuple[FilesystemArtifact, ...]:
    """Return artifacts with safe, unique filenames."""

    used: set[str] = set()
    normalized: list[FilesystemArtifact] = []
    for artifact in artifacts:
        name = unique_component(artifact.name, used)
        normalized.append(replace(artifact, name=name))
    return tuple(normalized)


def _coerce_artifacts(value: Any) -> tuple[FilesystemArtifact, ...]:
    if value is None:
        return ()
    if isinstance(value, FilesystemArtifact):
        return (value,)
    if isinstance(value, Mapping):
        if "name" in value and "content" in value:
            return (
                FilesystemArtifact(
                    name=str(value.get("name", "artifact.txt")),
                    content=str(value.get("content", "")),
                    description=str(value.get("description", "")),
                ),
            )
        return tuple(
            FilesystemArtifact(name=str(name), content=str(content))
            for name, content in value.items()
        )
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        artifacts: list[FilesystemArtifact] = []
        for item in value:
            artifacts.extend(_coerce_artifacts(item))
        return tuple(artifacts)
    return ()


def safe_component(value: str) -> str:
    """Return a filesystem-safe path component."""

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "default"


def unique_component(value: str, used: set[str]) -> str:
    """Return a safe path component that does not collide with prior components."""

    name = safe_component(value)
    if name not in used:
        used.add(name)
        return name
    stem, suffix = os.path.splitext(name)
    stem = stem or "artifact"
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1
