"""Small helpers for agentic proposal strategies."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

from simple_agent_lab.evolution.components.strategy import content_changing_edits
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import Proposal, Version


class RunnableAgent(Protocol):
    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]: ...


def consume_agent_run(agent: RunnableAgent, task: str, *, max_turns: int) -> None:
    """Run an agent to completion by consuming its lazy event iterator."""

    _state, events = agent.run(task, max_turns=max_turns)
    for _event in events:
        pass


def materialize_version_files(
    version: Version,
    root: Path,
    *,
    include: Callable[[str], bool] | None = None,
) -> None:
    """Write a version's files into ``root`` for an agent to inspect and edit."""

    for name in version.files():
        if include is not None and not include(name):
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(version.read(name), encoding="utf-8")


def changed_text_edits(base_tree: Path, changed_tree: Path) -> dict[str, str | None]:
    """Return full-text edits and tombstones between two text-only trees."""

    base_files = _text_file_map(base_tree)
    changed_files = _text_file_map(changed_tree)
    edits: dict[str, str | None] = {}
    for rel in sorted(set(base_files) | set(changed_files)):
        if rel not in changed_files:
            edits[rel] = None
            continue
        changed = changed_files[rel]
        if base_files.get(rel) != changed:
            edits[rel] = changed
    return edits


def surface_proposal_from_trees(
    *,
    base_version: Version,
    base_tree: Path,
    changed_tree: Path,
    surface: AgentSurface,
    editable_components: Sequence[str],
    base_hash: str,
    note: str,
    evidence: Sequence[str] = (),
    kind: str,
) -> Proposal:
    """Build a surface-validated proposal from a before/after workspace."""

    raw_edits = changed_text_edits(base_tree, changed_tree)
    validated = surface.validate_edits(raw_edits, components=editable_components)
    edits, unchanged = content_changing_edits(base_version, validated.edits)
    proposal_evidence = list(evidence)
    proposal_evidence.extend(
        f"discarded-disallowed-path:{path}" for path in validated.rejected
    )
    proposal_evidence.extend(f"discarded-unchanged-path:{path}" for path in unchanged)
    return Proposal(
        base=base_hash,
        edits=edits,
        note=note,
        evidence=tuple(proposal_evidence),
        kind=kind,
    )


def _text_file_map(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files
