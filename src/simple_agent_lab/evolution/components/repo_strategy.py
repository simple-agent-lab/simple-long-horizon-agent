"""Agentic source-tree meta-strategy helpers."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.evolution.source_tree import (
    SOURCE_ROOT,
    cheap_validate_source_tree,
    validate_source_tree_edits,
)
from simple_agent_lab.evolution.types import Context, Proposal

DEFAULT_SOURCE_TREE_AGENT_PROMPT = """You are a meta-agent improving Simple Agent Lab.

You may inspect and edit this temporary repository copy with bash. Make one
small, focused change under src/simple_agent_lab/ that is likely to improve the
current self-evolving agent system. Do not edit recipes or config migration in
this task.
"""

DEFAULT_SOURCE_TREE_AGENT_TASK = """Improve the source tree in this temporary copy.

Only changes under src/simple_agent_lab/ can become a proposal. Prefer small,
readable Python edits. When finished, reply with a short summary.
"""

_COPY_IGNORE_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


class _RunnableAgent(Protocol):
    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]: ...


AgentBuilder = Callable[..., _RunnableAgent]
ValidationFn = Callable[[Path, Mapping[str, str]], None]


def proposal_from_candidate_tree(
    base_tree: Path,
    changed_tree: Path,
    *,
    base_hash: str,
    note: str,
    evidence: Sequence[str] = (),
) -> Proposal:
    """Turn changed ``src/simple_agent_lab/**/*.py`` files into a Proposal.

    Deletions are recorded as evidence and ignored for now. The current
    ``candidate_source_artifacts`` staging helper overlays files but cannot
    faithfully tombstone inherited source files.
    """

    edits: dict[str, str] = {}
    proposal_evidence = list(evidence)

    for rel in _changed_paths(base_tree, changed_tree):
        rel_text = rel.as_posix()
        base_path = base_tree / rel
        changed_path = changed_tree / rel

        if not _is_under_source_root(rel):
            proposal_evidence.append(f"discarded-outside-source:{rel_text}")
            continue
        if changed_path.is_symlink() or base_path.is_symlink():
            proposal_evidence.append(f"discarded-symlink-source:{rel_text}")
            continue
        if not changed_path.exists():
            proposal_evidence.append(f"discarded-deleted-source:{rel_text}")
            continue
        if changed_path.suffix != ".py":
            proposal_evidence.append(f"discarded-non-python-source:{rel_text}")
            continue
        if not changed_path.is_file():
            continue

        try:
            content = changed_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            proposal_evidence.append(f"discarded-non-utf8-source:{rel_text}")
            continue
        errors = validate_source_tree_edits({rel_text: content})
        if errors:
            joined = "; ".join(errors)
            proposal_evidence.append(f"discarded-invalid-source:{rel_text}: {joined}")
            continue
        edits[rel_text] = content

    return Proposal(
        edits=edits,
        note=note,
        evidence=tuple(proposal_evidence),
        base=base_hash,
        kind="source",
    )


def source_tree_agent_strategy(
    *,
    provider: Any,
    repo_root: Path,
    max_turns: int = 20,
    validation: ValidationFn = cheap_validate_source_tree,
    parent_selection: str = "current",
    parent_selector: Callable[[Context, str], str] | None = None,
    agent_builder: AgentBuilder | None = None,
    system_prompt: str = DEFAULT_SOURCE_TREE_AGENT_PROMPT,
    task: str = DEFAULT_SOURCE_TREE_AGENT_TASK,
    name: str = "source_tree_meta_agent",
    surface: object | None = None,
    editable_components: Sequence[str] = (),
) -> Callable[[Context], Proposal | None]:
    """Return a strategy that lets a bash-capable meta-agent edit a repo copy."""

    root = Path(repo_root)
    build_agent = agent_builder or make_bash_agent
    _ = (surface, editable_components)

    def strategy(ctx: Context) -> Proposal | None:
        parent = _select_parent(
            ctx, parent_selection=parent_selection, parent_selector=parent_selector
        )
        base_version = ctx.version(parent)
        with tempfile.TemporaryDirectory(prefix="sal-source-tree-") as tmp:
            base_tree = Path(tmp) / "base"
            candidate = Path(tmp) / "candidate"
            shutil.copytree(
                root,
                base_tree,
                symlinks=True,
                ignore=_copy_ignore,
            )
            _overlay_source_version(base_tree, base_version.files(), base_version.read)
            shutil.copytree(base_tree, candidate, symlinks=True)

            agent = build_agent(
                provider=provider,
                cwd=candidate,
                name=name,
                system_prompt=system_prompt,
            )
            _state, events = agent.run(task, max_turns=max_turns)
            for _event in events:
                pass

            proposal = proposal_from_candidate_tree(
                base_tree,
                candidate,
                base_hash=parent,
                note="source-tree meta-agent edit",
                evidence=("source-tree-agent-ran",),
            )
            if not proposal.edits:
                return None

            files = {
                path: content
                for path, content in proposal.edits.items()
                if isinstance(content, str)
                and not validate_source_tree_edits({path: content})
            }
            if not files:
                return None
            try:
                validation(base_tree, files)
            except Exception:
                return None
            return proposal

    return strategy


def _select_parent(
    ctx: Context,
    *,
    parent_selection: str,
    parent_selector: Callable[[Context, str], str] | None,
) -> str:
    if parent_selection == "current":
        return ctx.current.hash
    if parent_selector is None:
        raise ValueError(
            "non-current parent selection requires a recipe-provided parent_selector"
        )
    return parent_selector(ctx, parent_selection) or ctx.current.hash


def _overlay_source_version(
    base_tree: Path,
    files: Sequence[str],
    read: Callable[[str], str],
) -> None:
    for path in files:
        rel = Path(path)
        if not _is_under_source_root(rel):
            continue
        target = base_tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_text(read(path), encoding="utf-8")


def _changed_paths(base_tree: Path, changed_tree: Path) -> list[Path]:
    base_files = _file_map(base_tree)
    changed_files = _file_map(changed_tree)
    paths = sorted(set(base_files) | set(changed_files))
    changed: list[Path] = []
    for rel in paths:
        base_path = base_files.get(rel)
        changed_path = changed_files.get(rel)
        if base_path is None or changed_path is None:
            changed.append(rel)
            continue
        if base_path.is_symlink() or changed_path.is_symlink():
            if base_path.resolve(strict=False) != changed_path.resolve(strict=False):
                changed.append(rel)
            continue
        if base_path.read_bytes() != changed_path.read_bytes():
            changed.append(rel)
    return changed


def _file_map(root: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if _skip_path(path):
            continue
        if path.is_file() or path.is_symlink():
            files[path.relative_to(root)] = path
    return files


def _skip_path(path: Path) -> bool:
    return any(part in _COPY_IGNORE_NAMES for part in path.parts)


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _COPY_IGNORE_NAMES}


def _is_under_source_root(path: Path) -> bool:
    parts = path.parts
    source_parts = Path(SOURCE_ROOT).parts
    return len(parts) > len(source_parts) and parts[: len(source_parts)] == source_parts
