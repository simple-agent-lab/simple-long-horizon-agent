"""Helpers for evolving code repositories as version artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from simple_agent_lab.evolution.types import Proposal

EditValue = str | bytes | None

DEFAULT_EXCLUDES = (".git/", "__pycache__/", ".pytest_cache/", ".ruff_cache/")


def directory_edits(
    root: str | Path, *, exclude: Iterable[str] = ()
) -> dict[str, str | bytes]:
    """Return full-file edits for every file under ``root``.

    Text files are decoded as UTF-8 for readability in manifests and tests;
    binary files stay as bytes so the version store can preserve them.
    """

    root_path = Path(root)
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(exclude)
    edits: dict[str, str | bytes] = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_path).as_posix()
        if _excluded(rel, excludes):
            continue
        data = path.read_bytes()
        try:
            edits[rel] = data.decode("utf-8")
        except UnicodeDecodeError:
            edits[rel] = data
    return edits


def proposal_from_changed_tree(
    base: str | Path,
    changed: str | Path,
    *,
    note: str = "",
    evidence: tuple[str, ...] = (),
    kind: str = "code",
    exclude: Iterable[str] = (),
) -> Proposal:
    """Compare two trees and produce full-file ``Proposal.edits``."""

    base_edits = directory_edits(base, exclude=exclude)
    changed_edits = directory_edits(changed, exclude=exclude)
    edits: dict[str, EditValue] = {}

    for rel, value in changed_edits.items():
        if base_edits.get(rel) != value:
            edits[rel] = value
    for rel in base_edits:
        if rel not in changed_edits:
            edits[rel] = None

    return Proposal(edits=edits, note=note, evidence=evidence, kind=kind)


def touched_paths(diff_text: str) -> tuple[str, ...]:
    """Return repository paths mentioned by ``diff --git`` headers."""

    paths = set()
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        old = _strip_diff_prefix(parts[2])
        new = _strip_diff_prefix(parts[3])
        paths.add(new if new != "/dev/null" else old)
    return tuple(sorted(paths))


def _excluded(rel: str, excludes: Iterable[str]) -> bool:
    for pattern in excludes:
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            if rel == prefix or rel.startswith(f"{prefix}/"):
                return True
        elif rel == pattern:
            return True
    return False


def _strip_diff_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path
