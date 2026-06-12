"""The version store: content-addressed immutable versions + pointer promotion.

This is kernel code (a guarantee, not a policy point). ``promote`` is the ONLY
mutation in the whole framework. Versions are never overwritten; rejected ones
are retained as stepping stones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from simple_agent_lab.evolution.types import (
    MANIFEST_NAME,
    Manifest,
    Version,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def version_hash(version_dir: Path) -> str:
    """sha256 over the sorted (relpath, file-sha256) walk, excluding the manifest."""

    parts: list[str] = []
    for path in sorted(version_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(version_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(f"{rel}:{digest}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def stage(
    workspace: Path,
    *,
    base: Version | None,
    edits: Mapping[str, str | bytes | None],
    manifest: Manifest | None = None,
) -> Version:
    """Copy ``base``, apply ``edits``, write the manifest, store under the hash.

    An edit value is full new content (``str`` text / ``bytes`` binary) or
    ``None`` (a tombstone removing an inherited file). Re-staging identical
    content returns the existing version with its ORIGINAL manifest.
    """

    scratch = workspace / "versions" / f".staging-{_now().replace(':', '')}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        if base is not None:
            for rel in base.files():
                src = base.dir / rel
                dst = scratch / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        for rel, value in edits.items():
            dst = scratch / rel
            if value is None:
                dst.unlink(missing_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, bytes):
                dst.write_bytes(value)
            else:
                dst.write_text(value, encoding="utf-8")

        digest = version_hash(scratch)
        final = workspace / "versions" / digest
        if final.exists():
            return Version(final)  # first provenance wins

        meta = manifest or Manifest()
        meta = Manifest(
            parent=meta.parent if meta.parent is not None else (base.hash if base else None),
            producer=meta.producer,
            evidence=meta.evidence,
            note=meta.note,
            created=meta.created or _now(),
            schema=meta.schema,
        )
        (scratch / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "parent": meta.parent,
                    "producer": meta.producer,
                    "evidence": list(meta.evidence),
                    "note": meta.note,
                    "created": meta.created,
                    "schema": meta.schema,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        scratch.rename(final)
        return Version(final)
    finally:
        if scratch.exists():
            for p in sorted(scratch.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            scratch.rmdir()


def _pointer_path(workspace: Path, *, namespace: str) -> Path:
    if namespace:
        return workspace / "pointers" / "shadow" / namespace / "current.json"
    return workspace / "pointers" / "current.json"


def current(workspace: Path, *, namespace: str = "") -> Version:
    path = _pointer_path(workspace, namespace=namespace)
    if not path.is_file():
        raise FileNotFoundError(f"no current version pointer at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return version(workspace, data["hash"])


def promote(workspace: Path, version_: Version, *, namespace: str = "") -> None:
    """Atomically point ``current`` at ``version_``. The only mutation primitive."""

    path = _pointer_path(workspace, namespace=namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.part")
    tmp.write_text(
        json.dumps({"hash": version_.hash, "updated": _now()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def version(workspace: Path, hash_: str) -> Version:
    return Version(workspace / "versions" / hash_)
