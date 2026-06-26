"""Semantic editable surfaces for self-evolving agents."""

from __future__ import annotations

import ast
import fnmatch
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from simple_agent_lab.evolution.types import Version


@dataclass(frozen=True)
class SurfaceComponent:
    id: str
    name: str
    description: str
    paths: tuple[str, ...]
    validators: tuple[str, ...] = ("path_allowed", "python_syntax")


@dataclass(frozen=True)
class ValidatedEdits:
    edits: dict[str, str | None]
    rejected: tuple[str, ...]


@dataclass(frozen=True)
class AgentSurface:
    id: str
    name: str
    description: str
    entrypoint: str
    default_files: Mapping[str, str]
    artifact_key: str
    components: tuple[SurfaceComponent, ...]
    excluded_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _relative_path(self.artifact_key, field="artifact_key")

    def seed_files(self) -> dict[str, str]:
        return dict(self.default_files)

    def component(self, id: str) -> SurfaceComponent:
        for component in self.components:
            if component.id == id:
                return component
        raise KeyError(f"unknown surface component {id!r}")

    def validate_edits(
        self,
        edits: Mapping[str, str | None],
        *,
        components: Sequence[str],
    ) -> ValidatedEdits:
        if not components:
            return ValidatedEdits({}, tuple(edits))

        allowed = tuple(self.component(name) for name in components)
        validators = {v for component in allowed for v in component.validators}
        out: dict[str, str | None] = {}
        rejected: list[str] = []
        for path, content in edits.items():
            if not _path_safe(path):
                rejected.append(path)
                continue
            if _path_excluded(path, self.excluded_paths):
                rejected.append(path)
                continue
            if not _path_allowed(path, allowed):
                rejected.append(path)
                continue
            if content is None:
                if "entrypoint_exists" in validators and path == self.entrypoint_path:
                    rejected.append(path)
                    continue
                out[path] = None
                continue
            if not isinstance(content, str):
                rejected.append(path)
                continue
            if "python_source" in validators and not path.endswith(".py"):
                rejected.append(path)
                continue
            if "python_syntax" in validators and path.endswith(".py"):
                if not _python_ok(content):
                    rejected.append(path)
                    continue
            if "entrypoint_exists" in validators and path == self.entrypoint_path:
                if not _entrypoint_ok(content, self.entrypoint_symbol):
                    rejected.append(path)
                    continue
            out[path] = content
        return ValidatedEdits(out, tuple(rejected))

    def prompt_brief(self, *, components: Sequence[str]) -> str:
        selected = [self.component(name) for name in components]
        lines = [
            f"Editable surface: {self.name}",
            self.description,
            "",
            f"Required entrypoint: {self.entrypoint}",
            "",
            "Editable components:",
        ]
        for component in selected:
            paths = ", ".join(component.paths)
            lines.append(f"- {component.id}: {component.description} Paths: {paths}")
        if self.excluded_paths:
            lines.extend(["", "Protected paths:"])
            for pattern in self.excluded_paths:
                lines.append(f"- {pattern}")
        return "\n".join(lines).strip()

    @property
    def entrypoint_path(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def entrypoint_symbol(self) -> str:
        parts = self.entrypoint.split(":", 1)
        return parts[1] if len(parts) == 2 else ""

    @property
    def artifact_root(self) -> str:
        parent = str(PurePosixPath(self.entrypoint_path).parent)
        return "" if parent == "." else f"{parent}/"

    def files_from_version(self, version: Version) -> dict[str, str]:
        files: dict[str, str] = {}
        patterns = tuple(
            path for component in self.components for path in component.paths
        )
        for name in version.files():
            if _path_excluded(name, self.excluded_paths):
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                files[name] = version.read(name)
        return files

    def artifacts_from_version(self, version: Version) -> dict[str, bytes]:
        payload = json.dumps(self.files_from_version(version), ensure_ascii=False)
        return {self.artifact_key: payload.encode("utf-8")}


def _path_allowed(path: str, components: Sequence[SurfaceComponent]) -> bool:
    if not _path_safe(path):
        return False
    patterns = tuple(pattern for component in components for pattern in component.paths)
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _path_excluded(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _path_safe(path: str) -> bool:
    return _relative_path_ok(path)


def _relative_path(path: str, *, field: str) -> str:
    if not _relative_path_ok(path):
        raise ValueError(
            f"{field} must be a non-empty relative path without '..': {path!r}"
        )
    return PurePosixPath(path).as_posix()


def _relative_path_ok(path: str) -> bool:
    pure_path = PurePosixPath(path)
    return (
        bool(path)
        and pure_path != PurePosixPath(".")
        and not (pure_path.is_absolute() or ".." in pure_path.parts)
    )


def _python_ok(content: str) -> bool:
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


def _entrypoint_ok(content: str, symbol: str) -> bool:
    if not symbol:
        return False
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
        for node in tree.body
    )
