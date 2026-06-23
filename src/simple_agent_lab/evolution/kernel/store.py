"""The version store: content-addressed immutable versions + pointer promotion.

This is kernel code (a guarantee, not a policy point). ``promote`` is the ONLY
mutation in the whole framework. Versions are never overwritten; rejected ones
are retained as stepping stones.

The store provides *atomic publish* (rename/replace make a new version or
pointer visible all-at-once), not crash durability: it does not fsync, so a
crash mid-write may lose the most recent unflushed bytes. That is an acceptable
trade for a lab; the invariant is that observers never see a half-written
version or pointer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from simple_agent_lab.evolution.types import (
    MANIFEST_NAME,
    Manifest,
    Version,
)

VERSION_HASH_RE = re.compile(r"^[0-9a-f]{12}$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_dest(scratch: Path, rel: str) -> Path:
    dst = (scratch / rel).resolve()
    root = scratch.resolve()
    if not (dst == root or root in dst.parents):
        raise ValueError(f"edit path escapes version dir: {rel!r}")
    return dst


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


def is_version_hash(value: str) -> bool:
    return bool(VERSION_HASH_RE.fullmatch(value))


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

    versions_root = workspace / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=versions_root, prefix=".staging-"))
    try:
        if base is not None:
            for rel in base.files():
                dst = _safe_dest(scratch, rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((base.dir / rel).read_bytes())
        for rel, value in edits.items():
            dst = _safe_dest(scratch, rel)
            if value is None:
                dst.unlink(missing_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, bytes):
                dst.write_bytes(value)
            else:
                dst.write_text(value, encoding="utf-8")

        digest = version_hash(scratch)
        final = versions_root / digest
        if final.exists():
            return Version(final)  # first provenance wins

        meta = manifest or Manifest()
        meta = Manifest(
            parent=meta.parent
            if meta.parent is not None
            else (base.hash if base else None),
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
        try:
            scratch.rename(final)
        except OSError:
            if final.exists():
                return Version(final)  # created concurrently; first provenance wins
            raise
        return Version(final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _pointer_path(workspace: Path, *, namespace: str) -> Path:
    # The shadow/<namespace>/ branch is a reserved Plan-2 seam (sandboxed/shadow
    # promotion); Plan 1 always uses the default empty namespace.
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
        json.dumps(
            {"hash": version_.hash, "updated": _now()}, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def version(workspace: Path, hash_: str) -> Version:
    if not is_version_hash(hash_):
        raise ValueError(f"not a valid version hash: {hash_!r}")
    versions_root = (workspace / "versions").resolve()
    path = (versions_root / hash_).resolve()
    if path.parent != versions_root:
        raise ValueError(f"version path escapes versions directory: {hash_!r}")
    return Version(path)
