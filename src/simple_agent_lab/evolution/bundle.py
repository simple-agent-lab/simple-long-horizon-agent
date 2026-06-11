"""Content-addressed behavior bundles: the unit of evolution.

A bundle is an immutable directory of behavior artifacts (provider config,
system prompt, playbook, lessons, skills). Its content hash is its version;
"current" is a pointer file. Promotion rewrites a pointer, rollback rewrites
it back, and nothing under ``bundles/`` is ever mutated or deleted — that is
the whole versioning story (see docs/design/20260610-evolution-framework-spec.md §2).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from simple_agent_lab.llm.provider import Provider


BUNDLE_SCHEMA = "simple-agent-lab.bundle.v1"
MANIFEST_NAME = "manifest.json"
PROVIDER_NAME = "provider.json"
HASH_LEN = 12

BundleLevel = str  # "task" | "meta"


@dataclass(frozen=True)
class Manifest:
    """Lineage and provenance *about* a bundle; excluded from its hash."""

    level: BundleLevel
    parent: str | None = None
    producer: str = ""
    evidence: tuple[str, ...] = ()
    note: str = ""
    created: str = ""
    schema: str = BUNDLE_SCHEMA


@dataclass(frozen=True)
class Bundle:
    """One agent version, as a typed read-only view over its directory.

    The directory stays the source of truth (``ls`` shows the same
    contract); this view only reads. Layout: ``manifest.json`` (lineage,
    excluded from the hash) plus any subset of behavior files —
    ``provider.json``, ``prompt.md``, ``playbook.md``, ``lessons.jsonl``,
    ``skills/``. ``dir`` is always exposed: anything not wrapped here is
    one ``open()`` away.
    """

    dir: Path

    @property
    def hash(self) -> str:
        return bundle_hash(self.dir)

    @property
    def manifest(self) -> Manifest:
        return read_manifest(self.dir)

    @property
    def parent(self) -> str | None:
        return self.manifest.parent

    def read(self, filename: str) -> str:
        """One behavior file's content ("" when absent)."""

        path = self.dir / filename
        return path.read_text() if path.exists() else ""

    def files(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                p.relative_to(self.dir).as_posix()
                for p in self.dir.rglob("*")
                if p.is_file() and p.name != MANIFEST_NAME
            )
        )


def bundle_hash(bundle_dir: Path) -> str:
    """Content hash over every file except the manifest.

    The manifest records lineage about the content (parent, note, producer);
    two bundles with identical behavior content but different notes must
    collide, so the manifest stays out of the hash.
    """

    digest = hashlib.sha256()
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()[:HASH_LEN]


def read_manifest(bundle_dir: Path) -> Manifest:
    raw = json.loads((bundle_dir / MANIFEST_NAME).read_text())
    raw.pop("schema", None)
    raw["evidence"] = tuple(raw.get("evidence", ()))
    return Manifest(**raw)


def load_provider(bundle_dir: Path) -> Provider | None:
    """The bundle's model config, or None when the bundle doesn't pin one."""

    path = bundle_dir / PROVIDER_NAME
    if not path.exists():
        return None
    return Provider(**json.loads(path.read_text()))


def stage_bundle(
    workspace: Path,
    *,
    manifest: Manifest,
    base: Bundle | None = None,
    edits: Mapping[str, str | bytes | None] | None = None,
) -> Bundle:
    """Create an immutable candidate bundle and return its typed view.

    Copies ``base`` (if given), applies ``edits``, then stores the result
    under ``bundles/<hash>/``. An edit value is the full new file content
    (``str`` for text, ``bytes`` for binary) or ``None`` — a tombstone that
    removes the file inherited from ``base`` (how a skill is retired).
    Staging the same content twice lands on the same directory — content
    addressing makes duplicate proposals visible for free.
    """

    tmp = workspace / "tmp" / uuid.uuid4().hex
    tmp.mkdir(parents=True)
    if base is not None:
        shutil.copytree(base.dir, tmp, dirs_exist_ok=True)
        (tmp / MANIFEST_NAME).unlink(missing_ok=True)
    for rel, content in (edits or {}).items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"edit path must be relative, got: {rel!r}")
        target = tmp / rel_path
        if content is None:  # tombstone: retire the inherited file
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)

    digest = bundle_hash(tmp)
    final = workspace / "bundles" / digest
    if final.exists():
        # Identical content already stored: reuse it and KEEP its manifest.
        # First provenance wins — rewriting lineage here would mutate the
        # archive, and rollback walks manifest parents.
        shutil.rmtree(tmp)
        return Bundle(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(final))
    created = manifest.created or datetime.now(timezone.utc).isoformat()
    record = asdict(manifest) | {
        "created": created,
        "evidence": list(manifest.evidence),
    }
    (final / MANIFEST_NAME).write_text(json.dumps(record, indent=2))
    return Bundle(final)


def _pointer_path(workspace: Path, pointer: str, namespace: str) -> Path:
    base = workspace / "pointers"
    if namespace:
        base = base / "shadow" / namespace
    return base / f"{pointer}.json"


def resolve(workspace: Path, pointer: str, *, namespace: str = "") -> Bundle:
    """Follow a pointer ("task" | "meta") to its current bundle."""

    path = _pointer_path(workspace, pointer, namespace)
    if not path.exists():
        raise FileNotFoundError(
            f"pointer {pointer!r} (namespace {namespace!r}) not set; "
            f"expected {path} — promote an initial bundle first"
        )
    digest = json.loads(path.read_text())["hash"]
    bundle_dir = workspace / "bundles" / digest
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"pointer {pointer!r} -> missing bundle {digest}")
    return Bundle(bundle_dir)


def promote(
    workspace: Path, pointer: str, bundle: Bundle, *, namespace: str = ""
) -> None:
    """Atomically point ``pointer`` at ``bundle`` — the only mutation."""

    path = _pointer_path(workspace, pointer, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "hash": bundle.hash,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
