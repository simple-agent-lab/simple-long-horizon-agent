"""Source-tree surfaces and artifact staging for self-evolution."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from simple_agent_lab.evolution.surface import AgentSurface, SurfaceComponent


SOURCE_ROOT = "src/simple_agent_lab"
CANDIDATE_TREE = "source_tree"
CANDIDATE_SRC = "source_tree/src"
CANDIDATE_PACKAGE = "source_tree/src/simple_agent_lab"
CANDIDATE_SOURCE_CONTAINER_SRC = "/agent/run/input/source_tree/src"

_SURFACE_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
_EDIT_EXTENSIONS = {".py"}
_MAX_SURFACE_BYTES = 32_000
_SKIP_PARTS = {
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so"}


def source_tree_surface(repo_root: Path) -> str:
    """Return a compact, readable view of editable package files."""

    source_root = repo_root / SOURCE_ROOT
    if not source_root.exists():
        raise ValueError(f"source root does not exist: {SOURCE_ROOT}")

    lines = [f"Source tree surface: {SOURCE_ROOT}", ""]
    for path in _walk_surface_files(source_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            data = path.read_bytes()
        except OSError as exc:
            lines.append(f"## {rel}")
            lines.append(f"<could not read: {exc}>")
            lines.append("")
            continue
        if len(data) > _MAX_SURFACE_BYTES:
            lines.append(f"## {rel}")
            lines.append(f"<skipped: {len(data)} bytes>")
            lines.append("")
            continue
        lines.append(f"## {rel}")
        lines.append(data.decode("utf-8", errors="replace").rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def source_tree_agent_surface(
    repo_root: Path,
    *,
    artifact_key: str = CANDIDATE_TREE,
    **_args: object,
) -> AgentSurface:
    """Return the config-facing surface for evolving the package source tree."""

    source_root = repo_root / SOURCE_ROOT
    if not source_root.exists():
        raise ValueError(f"source root does not exist: {SOURCE_ROOT}")

    default_files = {
        path.relative_to(repo_root).as_posix(): path.read_text(encoding="utf-8")
        for path in _walk_candidate_files(source_root)
        if path.suffix == ".py"
    }
    return AgentSurface(
        id="source_tree",
        name="Simple Agent Lab source tree",
        description="The framework source under src/simple_agent_lab/.",
        entrypoint=f"{SOURCE_ROOT}/__init__.py",
        default_files=default_files,
        artifact_key=artifact_key,
        components=(
            SurfaceComponent(
                id="everything",
                name="Whole framework source tree",
                description="All Python source files under src/simple_agent_lab/.",
                paths=(f"{SOURCE_ROOT}/**",),
                validators=("path_allowed", "python_syntax"),
            ),
        ),
    )


def validate_source_tree_edits(files: Mapping[str, str]) -> list[str]:
    """Return human-readable validation errors for proposed source edits."""

    errors: list[str] = []
    for path, content in files.items():
        errors.extend(_validate_source_tree_path(path))
        if not isinstance(content, str):
            errors.append(f"{path!r}: content must be a string")
    return errors


def candidate_source_artifacts(
    repo_root: Path,
    files: Mapping[str, str],
) -> dict[str, bytes]:
    """Stage the current package tree plus valid edits under ``source_tree/``."""

    errors = validate_source_tree_edits(files)
    if errors:
        raise ValueError("invalid source tree edits:\n" + "\n".join(errors))

    source_root = repo_root / SOURCE_ROOT
    if not source_root.exists():
        raise ValueError(f"source root does not exist: {SOURCE_ROOT}")

    artifacts: dict[str, bytes] = {}
    for path in _walk_candidate_files(source_root):
        rel = path.relative_to(source_root).as_posix()
        artifacts[f"{CANDIDATE_PACKAGE}/{rel}"] = path.read_bytes()

    for path, content in files.items():
        rel = _package_relative_path(path)
        artifacts[f"{CANDIDATE_PACKAGE}/{rel}"] = content.encode("utf-8")
    return artifacts


def cheap_validate_source_tree(repo_root: Path, files: Mapping[str, str]) -> None:
    """Stage a candidate tree in a temp dir and compile its Python package."""

    artifacts = candidate_source_artifacts(repo_root, files)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, data in artifacts.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        package = tmp_path / CANDIDATE_PACKAGE
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", str(package)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = f"compileall exited with {result.returncode}"
            raise RuntimeError(f"source tree compile check failed:\n{output}")


def _walk_surface_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in _walk_candidate_files(source_root):
        if path.suffix not in _SURFACE_EXTENSIONS:
            continue
        files.append(path)
    return files


def _walk_candidate_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(source_root).as_posix()
        if _skip_path(rel):
            continue
        files.append(path)
    return sorted(files)


def _validate_source_tree_path(path: str) -> list[str]:
    errors: list[str] = []
    if "\\" in path:
        errors.append(f"{path!r}: use forward-slash repo-relative paths")

    pure = PurePosixPath(path)
    if not path or pure == PurePosixPath("."):
        errors.append(f"{path!r}: path must be non-empty")
        return errors
    if pure.is_absolute():
        errors.append(f"{path!r}: absolute paths are not allowed")
    if ".." in pure.parts:
        errors.append(f"{path!r}: '..' traversal is not allowed")
    if not _is_under_source_root(pure):
        errors.append(f"{path!r}: path is outside src/simple_agent_lab/")
    if _skip_path(pure.as_posix()):
        errors.append(f"{path!r}: generated or cache paths are not allowed")
    if pure.suffix not in _EDIT_EXTENSIONS:
        errors.append(f"{path!r}: only .py source edits are allowed")
    return errors


def _is_under_source_root(path: PurePosixPath) -> bool:
    parts = path.parts
    source_parts = PurePosixPath(SOURCE_ROOT).parts
    return len(parts) > len(source_parts) and parts[: len(source_parts)] == source_parts


def _package_relative_path(path: str) -> str:
    pure = PurePosixPath(path)
    source_parts = PurePosixPath(SOURCE_ROOT).parts
    return PurePosixPath(*pure.parts[len(source_parts) :]).as_posix()


def _skip_path(path: str) -> bool:
    pure = PurePosixPath(path)
    if any(part in _SKIP_PARTS for part in pure.parts):
        return True
    return pure.suffix in _SKIP_SUFFIXES
