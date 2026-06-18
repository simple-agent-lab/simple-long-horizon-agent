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
            if "path_allowed" in validators and not _path_allowed(path, allowed):
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
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                rel = (
                    name[len(self.artifact_root) :]
                    if name.startswith(self.artifact_root)
                    else name
                )
                files[rel] = version.read(name)
        return files

    def artifacts_from_version(self, version: Version) -> dict[str, bytes]:
        payload = json.dumps(self.files_from_version(version), ensure_ascii=False)
        return {self.artifact_key: payload.encode("utf-8")}


def python_agent_surface(
    *,
    default_files: Mapping[str, str],
    artifact_key: str,
    version_root: str = "agent/",
) -> AgentSurface:
    root = _version_root(version_root)
    version_files = {root + path: text for path, text in default_files.items()}
    return AgentSurface(
        id="python_agent_package",
        name="Python agent package",
        description="A Python package that builds the benchmark-solving agent.",
        entrypoint=f"{root}agent_program.py:build_agent",
        default_files=version_files,
        artifact_key=artifact_key,
        components=(
            SurfaceComponent(
                id="agent_program",
                name="Agent program",
                description="Build-agent entrypoint and agent assembly.",
                paths=(f"{root}agent_program.py",),
                validators=("path_allowed", "python_syntax", "entrypoint_exists"),
            ),
            SurfaceComponent(
                id="prompts",
                name="Prompts",
                description="system prompts, task framing, and response policy.",
                paths=(f"{root}prompts.py", f"{root}prompts/**"),
            ),
            SurfaceComponent(
                id="tool_policy",
                name="Tool policy",
                description="Tool choice, shell usage, and retry behavior.",
                paths=(f"{root}tools.py", f"{root}tool_policy.py", f"{root}tools/**"),
            ),
            SurfaceComponent(
                id="memory_policy",
                name="Memory policy",
                description="How prior evidence or notes are used.",
                paths=(f"{root}memory.py", f"{root}memory/**"),
            ),
            SurfaceComponent(
                id="everything",
                name="Whole agent package",
                description="Unrestricted edits to the whole agent package.",
                paths=(f"{root}**",),
                validators=("path_allowed", "python_syntax", "entrypoint_exists"),
            ),
        ),
    )


def _path_allowed(path: str, components: Sequence[SurfaceComponent]) -> bool:
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        return False
    patterns = tuple(pattern for component in components for pattern in component.paths)
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _version_root(version_root: str) -> str:
    root = version_root.rstrip("/")
    path = PurePosixPath(root)
    if not root or root == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"version_root must be a non-empty relative path without '..': {version_root!r}"
        )
    return f"{path.as_posix()}/"


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
