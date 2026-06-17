# Evolution Infra — Implementation Plan 1: Substrate + Human-Function Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the kernel (`store`/`log`/`loop`) and the swappable components (`rollout`/`reward`/`criterion`) plus `registry`/`config`/`experiment`, so a researcher can run an evolution experiment end-to-end with a human-written strategy function on the fake backend.

**Architecture:** A fixed kernel owns the guarantees (immutable content-addressed versions, append-only decision log, the ~20-line loop driver). Swappable components own all policy and are plain callables. The `Experiment` object only wires components together; no policy lives in it. See spec: `docs/design/20260612-evolution-infra-redesign.md`.

**Tech Stack:** Python 3.10+, stdlib only (`hashlib`, `json`, `dataclasses`, `pathlib`, `concurrent.futures`), `unittest` for tests. Reuses `simple_agent_lab.evals` (`run_dataset`, `FakeBackend`, `LocalDirStore`, `RESULT_KEY`, `TRACE_KEY`) and `simple_agent_lab.trace.jsonl` (`write_jsonl_atomic`, `read_jsonl`).

**Scope note:** The LLM-agent `strategy` and the example strategy library are deferred to Plan 2. Plan 1 ships only human-written strategies, which is enough to satisfy all four spec acceptance criteria.

---

## File Structure

Create under `src/simple_agent_lab/evolution/`:

```text
evolution/
  __init__.py            # public exports (~12 names)
  types.py               # Version, Run, Slice, Proposal, Verdict, Decision, Context, Manifest, RunScores
  kernel/
    __init__.py
    store.py             # version_hash, stage, current, promote, version
    log.py               # append, read, hit_rate
    loop.py              # score, means, step, run
  components/
    __init__.py
    reward.py            # result_key (default), cost_tokens
    criterion.py         # improve, not_worse, guarded
    rollout.py           # dataset_rollout
  registry.py            # ROLLOUTS/REWARDS/STRATEGIES/CRITERIA dicts + Use + build
  config.py              # Config, Use
  experiment.py          # Experiment (+ from_config)
```

Create under `tests/unit/`:

```text
test_evolution_types.py
test_evolution_store.py
test_evolution_log.py
test_evolution_criterion.py
test_evolution_reward.py
test_evolution_loop.py
test_evolution_rollout.py
test_evolution_experiment.py
```

**Import hygiene:** all pure data views live in `types.py` (no intra-package deps), so `kernel/*` and `components/*` import from `evolution.types` and never from each other except `loop.py` → `store`/`log`. No circular imports.

Run a single test with:

```bash
uv run python -m unittest tests.unit.test_evolution_store.StoreTest.test_stage_is_content_addressed
```

Run the whole suite with:

```bash
uv run python -m unittest discover -s tests/unit
```

---

## Task 1: Core types (`types.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/types.py`
- Test: `tests/unit/test_evolution_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_types.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.types import Run, Slice, Version


class TypesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_slice_sha_is_stable_and_order_independent(self) -> None:
        a = Slice("s", ({"instance_id": "b"}, {"instance_id": "a"}))
        b = Slice("s", ({"instance_id": "a"}, {"instance_id": "b"}))
        self.assertEqual(a.sha, b.sha)
        self.assertEqual(len(a.sha), 12)

    def test_run_reads_result_and_reward(self) -> None:
        run_dir = self.tmp / "r1" / "i1"
        (run_dir / "out").mkdir(parents=True)
        (run_dir / "out" / "result.json").write_text(json.dumps({"reward": 0.5}))
        run = Run(run_dir)
        self.assertTrue(run.ok)
        self.assertEqual(run.instance_id, "i1")
        self.assertEqual(run.reward, 0.5)
        self.assertEqual(run.result["reward"], 0.5)

    def test_run_missing_result_is_not_ok(self) -> None:
        run_dir = self.tmp / "r1" / "i2"
        run_dir.mkdir(parents=True)
        run = Run(run_dir)
        self.assertFalse(run.ok)
        self.assertIsNone(run.reward)

    def test_version_reads_files(self) -> None:
        vdir = self.tmp / "versions" / "abc"
        vdir.mkdir(parents=True)
        (vdir / "prompt.md").write_text("hello")
        v = Version(vdir)
        self.assertEqual(v.hash, "abc")
        self.assertEqual(v.read("prompt.md"), "hello")
        self.assertEqual(v.read("missing.md"), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_types -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent_lab.evolution'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/types.py
"""The complete vocabulary of the evolution framework: five read-only views
plus three small supporting types. Views wrap a directory and parse lazily;
the directory is the source of truth and ``.dir`` is always the escape hatch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Aggregated/per-run score shapes shared by reward + criterion.
DimScores = Mapping[str, float]                      # one run: {dim: value}
RunScores = Mapping[str, DimScores]                  # {instance_id: {dim: value}}

BUNDLE_SCHEMA = "simple-agent-lab.version.v1"
MANIFEST_NAME = "manifest.json"
PROVIDER_NAME = "provider.json"


@dataclass(frozen=True)
class Manifest:
    """Provenance recorded alongside a version's content (never hashed)."""

    parent: str | None = None
    producer: str = ""
    evidence: tuple[str, ...] = ()
    note: str = ""
    created: str = ""
    schema: str = BUNDLE_SCHEMA


@dataclass(frozen=True)
class Version:
    """An immutable agent version: a directory of artifact files + a manifest."""

    dir: Path

    @property
    def hash(self) -> str:
        return self.dir.name

    @property
    def manifest(self) -> Manifest:
        path = self.dir / MANIFEST_NAME
        if not path.is_file():
            return Manifest()
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(
            parent=data.get("parent"),
            producer=data.get("producer", ""),
            evidence=tuple(data.get("evidence", ())),
            note=data.get("note", ""),
            created=data.get("created", ""),
            schema=data.get("schema", BUNDLE_SCHEMA),
        )

    @property
    def parent(self) -> str | None:
        return self.manifest.parent

    def read(self, filename: str) -> str:
        path = self.dir / filename
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def files(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(p.relative_to(self.dir))
                for p in self.dir.rglob("*")
                if p.is_file() and p.name != MANIFEST_NAME
            )
        )


@dataclass(frozen=True)
class Run:
    """One task instance executed: a view over its run directory."""

    dir: Path

    @property
    def instance_id(self) -> str:
        return self.dir.name

    @property
    def run_id(self) -> str:
        return self.dir.parent.name

    @property
    def ref(self) -> str:
        return f"{self.run_id}/{self.instance_id}"

    @property
    def ok(self) -> bool:
        return (self.dir / "out" / "result.json").is_file()

    @property
    def result(self) -> Mapping[str, Any]:
        path = self.dir / "out" / "result.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def reward(self) -> float | None:
        value = self.result.get("reward")
        return float(value) if value is not None else None

    def events(self) -> tuple[Mapping[str, Any], ...]:
        path = self.dir / "out" / "trajectory.jsonl"
        if not path.is_file():
            return ()
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return ()
        try:
            record = json.loads(lines[-1])
        except json.JSONDecodeError:
            return ()
        return tuple(record.get("events", ()))


@dataclass(frozen=True)
class Slice:
    """The frozen instance set used for a fair A/B comparison."""

    id: str
    instances: tuple[Mapping[str, Any], ...] = ()

    @property
    def sha(self) -> str:
        ids = sorted(
            str(inst.get("instance_id", n)) for n, inst in enumerate(self.instances)
        )
        return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()[:12]

    @property
    def n(self) -> int:
        return len(self.instances)


@dataclass(frozen=True)
class Proposal:
    """A strategy's output: one candidate change with provenance."""

    edits: Mapping[str, str | bytes | None]  # path -> content; None retires the file
    note: str = ""
    evidence: tuple[str, ...] = ()
    base: str = ""  # version hash to branch from; "" = the current version
    kind: str = ""  # free-form tag for analytics ("lesson", "prompt", ...)


@dataclass(frozen=True)
class Verdict:
    """A criterion's output."""

    accepted: bool
    reason: str
    deltas: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """One logged comparison."""

    id: str
    ts: str
    baseline: Mapping[str, Any]
    candidate: Mapping[str, Any]
    slice: Mapping[str, Any]
    accepted: bool
    reason: str
    deltas: Mapping[str, float] = field(default_factory=dict)
    runs: Mapping[str, str] = field(default_factory=dict)
    kind: str = ""
    schema: str = "simple-agent-lab.decision.v1"

    @property
    def outcome(self) -> str:
        return "accepted" if self.accepted else "rejected"


RewardFn = Callable[[Run], "float | Mapping[str, float]"]


@dataclass(frozen=True)
class Context:
    """What a strategy sees each step. All typed views; no raw paths."""

    runs: tuple[Run, ...]
    current: Version
    workspace: Path
    decisions: tuple[Decision, ...] = ()
    reward: RewardFn = lambda run: run.reward if run.reward is not None else 0.0

    @property
    def failures(self) -> tuple[Run, ...]:
        def score(run: Run) -> float:
            value = self.reward(run)
            if isinstance(value, Mapping):
                return float(value.get("reward", 0.0))
            return float(value)

        return tuple(r for r in self.runs if score(r) <= 0.0)

    def version(self, hash_: str) -> Version:
        return Version(self.workspace / "versions" / hash_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_types -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/types.py tests/unit/test_evolution_types.py
git commit -m "feat(evolution): core types (Version, Run, Slice, Proposal, Decision, Context)"
```

---

## Task 2: Version store (`kernel/store.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/kernel/__init__.py` (empty)
- Create: `src/simple_agent_lab/evolution/kernel/store.py`
- Test: `tests/unit/test_evolution_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_store.py
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Manifest


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)

    def test_stage_is_content_addressed(self) -> None:
        a = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})
        b = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})
        self.assertEqual(a.hash, b.hash)  # identical content -> same hash
        c = store.stage(self.ws, base=None, edits={"prompt.md": "bye"})
        self.assertNotEqual(a.hash, c.hash)

    def test_stage_applies_edits_over_base_and_tombstones(self) -> None:
        base = store.stage(self.ws, base=None, edits={"prompt.md": "p", "old.md": "x"})
        child = store.stage(
            self.ws, base=base, edits={"prompt.md": "p2", "old.md": None}
        )
        self.assertEqual(child.read("prompt.md"), "p2")
        self.assertEqual(child.read("old.md"), "")  # tombstoned
        self.assertEqual(child.parent, base.hash)

    def test_promote_current_rollback(self) -> None:
        a = store.stage(self.ws, base=None, edits={"prompt.md": "a"})
        store.promote(self.ws, a)
        self.assertEqual(store.current(self.ws).hash, a.hash)
        b = store.stage(self.ws, base=a, edits={"prompt.md": "b"})
        store.promote(self.ws, b)
        self.assertEqual(store.current(self.ws).hash, b.hash)
        store.promote(self.ws, store.version(self.ws, b.parent))  # rollback
        self.assertEqual(store.current(self.ws).hash, a.hash)

    def test_restage_preserves_original_manifest(self) -> None:
        first = store.stage(self.ws, base=None, edits={"p": "x"}, manifest=Manifest(note="first"))
        again = store.stage(self.ws, base=None, edits={"p": "x"}, manifest=Manifest(note="second"))
        self.assertEqual(first.hash, again.hash)
        self.assertEqual(again.manifest.note, "first")  # first provenance wins

    def test_current_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            store.current(self.ws)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_store -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent_lab.evolution.kernel'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/kernel/__init__.py
```

```python
# src/simple_agent_lab/evolution/kernel/store.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_store -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/kernel/__init__.py src/simple_agent_lab/evolution/kernel/store.py tests/unit/test_evolution_store.py
git commit -m "feat(evolution): content-addressed version store with promote/rollback"
```

---

## Task 3: Decision log (`kernel/log.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/kernel/log.py`
- Test: `tests/unit/test_evolution_log.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_log.py
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import log
from simple_agent_lab.evolution.types import Slice, Verdict


class LogTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)

    def _append(self, accepted: bool, kind: str) -> None:
        log.append(
            self.ws,
            baseline={"hash": "aaa", "scores": {"reward": 0.3}},
            candidate={"hash": "bbb", "parent": "aaa", "scores": {"reward": 0.5}},
            slice_=Slice("demo", ({"instance_id": "i1"},)),
            verdict=Verdict(accepted, "because", {"reward": 0.2}),
            kind=kind,
            runs={"baseline": "r-a", "candidate": "r-b"},
        )

    def test_append_assigns_id_and_reads_back(self) -> None:
        self._append(True, "prompt")
        self._append(False, "lesson")
        rows = log.read(self.ws)
        self.assertEqual([d.id for d in rows], ["d-000001", "d-000002"])
        self.assertEqual(rows[0].accepted, True)
        self.assertEqual(rows[0].kind, "prompt")
        self.assertEqual(rows[0].deltas["reward"], 0.2)
        self.assertEqual(rows[0].runs["candidate"], "r-b")

    def test_read_filters_and_hit_rate(self) -> None:
        self._append(True, "prompt")
        self._append(False, "prompt")
        self._append(True, "prompt")
        self.assertEqual(len(log.read(self.ws, kind="prompt")), 3)
        self.assertAlmostEqual(log.hit_rate(self.ws, kind="prompt"), 2 / 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_log -v`
Expected: FAIL with `ImportError: cannot import name 'log'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/kernel/log.py
"""The decision log: one append-only JSONL record per comparison.

Kernel code. Records carry workspace-relative run refs (the log outlives any
machine). No candidate can edit this file; only the loop appends to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.types import Decision, Slice, Verdict
from simple_agent_lab.trace.jsonl import read_jsonl, write_jsonl

LOG_NAME = "decisions.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(workspace: Path) -> Path:
    return workspace / LOG_NAME


def _to_record(decision: Decision) -> dict[str, Any]:
    return {
        "schema": decision.schema,
        "id": decision.id,
        "ts": decision.ts,
        "kind": decision.kind,
        "baseline": dict(decision.baseline),
        "candidate": dict(decision.candidate),
        "slice": dict(decision.slice),
        "accepted": decision.accepted,
        "reason": decision.reason,
        "deltas": dict(decision.deltas),
        "runs": dict(decision.runs),
    }


def _from_record(record: Mapping[str, Any]) -> Decision:
    return Decision(
        id=record["id"],
        ts=record.get("ts", ""),
        baseline=record.get("baseline", {}),
        candidate=record.get("candidate", {}),
        slice=record.get("slice", {}),
        accepted=bool(record.get("accepted", False)),
        reason=record.get("reason", ""),
        deltas=record.get("deltas", {}),
        runs=record.get("runs", {}),
        kind=record.get("kind", ""),
        schema=record.get("schema", "simple-agent-lab.decision.v1"),
    )


def read(
    workspace: Path,
    *,
    kind: str | None = None,
    accepted: bool | None = None,
    limit: int | None = None,
) -> list[Decision]:
    path = _path(workspace)
    if not path.is_file():
        return []
    rows = [_from_record(r) for r in read_jsonl(path)]
    if kind is not None:
        rows = [d for d in rows if d.kind == kind]
    if accepted is not None:
        rows = [d for d in rows if d.accepted == accepted]
    if limit is not None:
        rows = rows[-limit:]
    return rows


def append(
    workspace: Path,
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    slice_: Slice,
    verdict: Verdict,
    kind: str = "",
    runs: Mapping[str, str] | None = None,
) -> Decision:
    existing = read(workspace)
    decision = Decision(
        id=f"d-{len(existing) + 1:06d}",
        ts=_now(),
        baseline=dict(baseline),
        candidate=dict(candidate),
        slice={"id": slice_.id, "sha": slice_.sha, "n": slice_.n},
        accepted=verdict.accepted,
        reason=verdict.reason,
        deltas=dict(verdict.deltas),
        runs=dict(runs or {}),
        kind=kind,
    )
    records = [_to_record(d) for d in existing] + [_to_record(decision)]
    write_jsonl(_path(workspace), records)
    return decision


def hit_rate(
    workspace: Path, *, kind: str | None = None, window: int | None = None
) -> float | None:
    rows = read(workspace, kind=kind, limit=window)
    if not rows:
        return None
    return sum(1 for d in rows if d.accepted) / len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_log -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/kernel/log.py tests/unit/test_evolution_log.py
git commit -m "feat(evolution): append-only decision log"
```

---

## Task 4: Criterion component (`components/criterion.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/components/__init__.py` (empty)
- Create: `src/simple_agent_lab/evolution/components/criterion.py`
- Test: `tests/unit/test_evolution_criterion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_criterion.py
from __future__ import annotations

import unittest

from simple_agent_lab.evolution.components.criterion import (
    guarded,
    improve,
    not_worse,
)


class CriterionTest(unittest.TestCase):
    def test_improve_accepts_strict_gain(self) -> None:
        base = {"i1": {"reward": 0.0}, "i2": {"reward": 0.0}}
        cand = {"i1": {"reward": 1.0}, "i2": {"reward": 0.0}}
        v = improve("reward")(base, cand)
        self.assertTrue(v.accepted)
        self.assertAlmostEqual(v.deltas["reward"], 0.5)

    def test_improve_rejects_no_gain(self) -> None:
        base = {"i1": {"reward": 0.5}}
        cand = {"i1": {"reward": 0.5}}
        self.assertFalse(improve("reward")(base, cand).accepted)

    def test_improve_missing_dimension_raises(self) -> None:
        with self.assertRaises(KeyError):
            improve("nope")({"i1": {"reward": 0.1}}, {"i1": {"reward": 0.2}})

    def test_not_worse_guard(self) -> None:
        base = {"i1": {"reward": 1.0}}
        cand = {"i1": {"reward": 0.99}}
        self.assertTrue(not_worse("reward", tol=0.05)(base, cand).accepted)
        self.assertFalse(not_worse("reward", tol=0.0)(base, cand).accepted)

    def test_guarded_requires_objective_and_all_guards(self) -> None:
        crit = guarded(improve("reward"), [not_worse("safety")])
        base = {"i1": {"reward": 0.0, "safety": 1.0}}
        good = {"i1": {"reward": 1.0, "safety": 1.0}}
        bad = {"i1": {"reward": 1.0, "safety": 0.0}}
        self.assertTrue(crit(base, good).accepted)
        self.assertFalse(crit(base, bad).accepted)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_criterion -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent_lab.evolution.components'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/components/__init__.py
```

```python
# src/simple_agent_lab/evolution/components/criterion.py
"""Criterion combinators: how to judge candidate vs baseline.

A criterion is ``(RunScores, RunScores) -> Verdict``. Aggregation lives HERE
(each criterion receives per-run scores for both sides), so paired and
multi-dimensional judgments are possible without a separate measures system.
Declarative combinators are chosen over weighted sums because constraint-style
judgments produce auditable reasons in the decision log.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from simple_agent_lab.evolution.types import RunScores, Verdict

Criterion = Callable[[RunScores, RunScores], Verdict]


def _mean(scores: RunScores, dim: str) -> float:
    values = []
    for per_dim in scores.values():
        if dim not in per_dim:
            raise KeyError(
                f"criterion reads dimension {dim!r} but a run's scores have only "
                f"{sorted(per_dim)} — check the reward function"
            )
        values.append(per_dim[dim])
    return sum(values) / len(values) if values else 0.0


def improve(dim: str = "reward", *, min_delta: float = 0.0) -> Criterion:
    """Accept when the mean of ``dim`` strictly climbs (by at least ``min_delta``)."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        b, c = _mean(baseline, dim), _mean(candidate, dim)
        delta = c - b
        accepted = delta > 0 if min_delta == 0 else delta >= min_delta
        return Verdict(
            accepted,
            f"{dim} {b:.4g}->{c:.4g} (delta {delta:+.4g}, min_delta {min_delta})",
            {dim: delta},
        )

    return judge


def not_worse(dim: str = "reward", *, tol: float = 0.0) -> Criterion:
    """A guard: accept when ``dim`` does not drop by more than ``tol``."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        b, c = _mean(baseline, dim), _mean(candidate, dim)
        delta = c - b
        accepted = c >= b - tol
        word = "ok" if accepted else "regressed"
        return Verdict(accepted, f"guard {dim} {word} ({delta:+.4g}, tol {tol})", {dim: delta})

    return judge


def guarded(objective: Criterion, guards: Sequence[Criterion]) -> Criterion:
    """Optimize ``objective`` subject to every guard holding ("X subject to Y")."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        obj = objective(baseline, candidate)
        deltas = dict(obj.deltas)
        reasons = [obj.reason]
        accepted = obj.accepted
        for guard in guards:
            g = guard(baseline, candidate)
            deltas.update(g.deltas)
            reasons.append(g.reason)
            accepted = accepted and g.accepted
        return Verdict(accepted, "; ".join(reasons), deltas)

    return judge
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_criterion -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/components/__init__.py src/simple_agent_lab/evolution/components/criterion.py tests/unit/test_evolution_criterion.py
git commit -m "feat(evolution): declarative criterion combinators (improve/not_worse/guarded)"
```

---

## Task 5: Reward component (`components/reward.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/components/reward.py`
- Test: `tests/unit/test_evolution_reward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_reward.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.components.reward import cost_tokens, result_key
from simple_agent_lab.evolution.types import Run


class RewardTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _run(self, *, result: dict, events: list[dict] | None = None) -> Run:
        d = self.tmp / "r" / "i1"
        (d / "out").mkdir(parents=True)
        (d / "out" / "result.json").write_text(json.dumps(result))
        if events is not None:
            (d / "out" / "trajectory.jsonl").write_text(json.dumps({"events": events}))
        return Run(d)

    def test_result_key_reads_reward(self) -> None:
        self.assertEqual(result_key(self._run(result={"reward": 0.7})), 0.7)

    def test_result_key_crash_is_zero(self) -> None:
        crashed = Run(self.tmp / "r" / "missing")
        self.assertEqual(result_key(crashed), 0.0)

    def test_cost_tokens_sums_usage(self) -> None:
        run = self._run(
            result={"reward": 1.0},
            events=[
                {"usage": {"input_tokens": 10, "output_tokens": 5}},
                {"usage": {"input_tokens": 20, "output_tokens": 0}},
                {"kind": "other"},
            ],
        )
        self.assertEqual(cost_tokens(run), 35.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_reward -v`
Expected: FAIL with `ImportError: cannot import name 'result_key'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/components/reward.py
"""Reward functions: how to score one run.

A reward is ``(Run) -> float | Mapping[str, float]``. Returning a float means
the single ``reward`` dimension; returning a dict declares multiple dimensions.
This is the ONLY scoring surface — there is no separate "measures" system.
"""

from __future__ import annotations

from simple_agent_lab.evolution.types import Run


def result_key(run: Run) -> float:
    """The default: the ``result.json`` reward key; a crashed run scores 0."""

    return run.reward if run.reward is not None else 0.0


def cost_tokens(run: Run) -> float:
    """Total input+output tokens summed across the run's usage events."""

    total = 0.0
    for event in run.events():
        usage = event.get("usage")
        if isinstance(usage, dict):
            total += float(usage.get("input_tokens") or 0)
            total += float(usage.get("output_tokens") or 0)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_reward -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/components/reward.py tests/unit/test_evolution_reward.py
git commit -m "feat(evolution): reward functions (result_key default, cost_tokens)"
```

---

## Task 6: Loop driver (`kernel/loop.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/kernel/loop.py`
- Test: `tests/unit/test_evolution_loop.py`

The loop wires components and runs the `observe -> propose -> compare -> record` step. `score()` converts runs to `RunScores` via the reward; `means()` aggregates per dimension for the log.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_loop.py
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.kernel import loop, store
from simple_agent_lab.evolution.types import Context, Proposal, Run, Slice, Version


@dataclass(frozen=True)
class Components:
    rollout: Callable[[Version, Slice], Sequence[Run]]
    reward: Callable[[Run], float]
    strategy: Callable[[Context], "Proposal | None"]
    criterion: Callable


def stub_rollout(rewards_by_prompt: dict[str, float], runs_root: Path):
    def rollout(version: Version, slice_: Slice) -> list[Run]:
        reward = rewards_by_prompt[version.read("prompt.md")]
        run_dir = runs_root / f"{version.hash}-{slice_.sha}" / "i1"
        (run_dir / "out").mkdir(parents=True, exist_ok=True)
        (run_dir / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [Run(run_dir)]

    return rollout


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)
        initial = store.stage(self.ws, base=None, edits={"prompt.md": "weak"})
        store.promote(self.ws, initial)
        self.slice = Slice("demo", ({"instance_id": "i1"},))

    def _components(self, strategy) -> Components:
        return Components(
            rollout=stub_rollout({"weak": 0.3, "strong": 0.7}, self.ws / "runs"),
            reward=result_key,
            strategy=strategy,
            criterion=improve("reward"),
        )

    def test_accepted_proposal_promotes(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, note="try strong", kind="prompt")

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertTrue(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "strong")
        self.assertEqual(decision.deltas["reward"], 0.4)

    def test_rejected_proposal_keeps_current(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "weak"}, note="no change", kind="prompt")

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertFalse(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "weak")

    def test_no_proposal_returns_none(self) -> None:
        decision = loop.step(self.ws, self._components(lambda ctx: None), self.slice)
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_loop -v`
Expected: FAIL with `ImportError: cannot import name 'loop'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/kernel/loop.py
"""The loop driver: observe -> propose -> compare -> record.

Kernel code. The "gate" is not a separate noun — it is this sequence (two
rollouts + apply criterion + append to the log). Promotion is host-side and
evidence-driven, so the same guarantee holds whether the strategy is a human
function or (Plan 2) an LLM agent.

``Components`` is any object exposing ``rollout``, ``reward``, ``strategy``,
``criterion`` attributes (the ``Experiment`` provides one; tests use a small
dataclass).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Manifest,
    Run,
    RunScores,
    Slice,
)


class Components(Protocol):
    rollout: Any
    reward: Any
    strategy: Any
    criterion: Any


def score(runs: Sequence[Run], reward: Any) -> dict[str, dict[str, float]]:
    """Apply ``reward`` to each run, normalizing to {instance_id: {dim: value}}."""

    out: dict[str, dict[str, float]] = {}
    for run in runs:
        value = reward(run)
        dims = value if isinstance(value, Mapping) else {"reward": float(value)}
        out[run.instance_id] = {k: float(v) for k, v in dims.items()}
    return out


def means(run_scores: RunScores) -> dict[str, float]:
    """Per-dimension mean over runs — the aggregate recorded in the log."""

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for per_dim in run_scores.values():
        for dim, val in per_dim.items():
            sums[dim] = sums.get(dim, 0.0) + val
            counts[dim] = counts.get(dim, 0) + 1
    return {dim: sums[dim] / counts[dim] for dim in sums}


def step(
    workspace: Path,
    components: Components,
    slice_: Slice,
    *,
    auto_promote: bool = True,
) -> Decision | None:
    current = store.current(workspace)
    base_runs = components.rollout(current, slice_)
    proposal = components.strategy(
        Context(
            runs=tuple(base_runs),
            current=current,
            workspace=workspace,
            decisions=tuple(log.read(workspace)),
            reward=components.reward,
        )
    )
    if proposal is None:
        return None

    base = (
        store.version(workspace, proposal.base)
        if proposal.base
        else current
    )
    candidate = store.stage(
        workspace,
        base=base,
        edits=proposal.edits,
        manifest=Manifest(
            parent=base.hash,
            producer=getattr(components.strategy, "__name__", "strategy"),
            evidence=proposal.evidence,
            note=proposal.note,
        ),
    )
    cand_runs = components.rollout(candidate, slice_)

    base_scores = score(base_runs, components.reward)
    cand_scores = score(cand_runs, components.reward)
    verdict = components.criterion(base_scores, cand_scores)

    decision = log.append(
        workspace,
        baseline={"hash": current.hash, "scores": means(base_scores)},
        candidate={
            "hash": candidate.hash,
            "parent": candidate.parent,
            "scores": means(cand_scores),
            "note": candidate.manifest.note,
            "evidence": list(candidate.manifest.evidence),
        },
        slice_=slice_,
        verdict=verdict,
        kind=proposal.kind,
        runs={
            "baseline": _run_id(base_runs),
            "candidate": _run_id(cand_runs),
        },
    )
    if verdict.accepted and auto_promote:
        store.promote(workspace, candidate)
    return decision


def run(
    workspace: Path,
    components: Components,
    slice_: Slice,
    *,
    n: int = 1,
    auto_promote: bool = True,
) -> list[Decision]:
    out = []
    for _ in range(n):
        decision = step(workspace, components, slice_, auto_promote=auto_promote)
        if decision is not None:
            out.append(decision)
    return out


def _run_id(runs: Sequence[Run]) -> str:
    return runs[0].run_id if runs else ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_loop -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/kernel/loop.py tests/unit/test_evolution_loop.py
git commit -m "feat(evolution): loop driver (step/run) wiring components on the kernel"
```

---

## Task 7: Rollout component (`components/rollout.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/components/rollout.py`
- Test: `tests/unit/test_evolution_rollout.py`

`dataset_rollout` wraps the existing eval harness. It uses a **deterministic** `run_id = f"{version.hash}-{slice.sha}"`, so re-running the same `(version, slice)` reuses the existing run dir (measurement reuse) rather than re-rolling. The factory binds deployment concerns (suite/backend/store); the slice arrives per call.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_rollout.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import RESULT_KEY, TRACE_KEY, FakeBackend, LocalDirStore
from simple_agent_lab.evals.protocols import LaunchSpec, RunSpec
from simple_agent_lab.evolution.components.rollout import dataset_rollout
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Slice


class _DemoSuite:
    name = "demo"
    container_module = "demo.container"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="demo:latest", workdir="/work")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


def _simulate(reward: float):
    def on_run(spec: RunSpec, bound) -> None:
        bound.put(TRACE_KEY, b'{"events": []}\n')
        bound.put(RESULT_KEY, (json.dumps({"reward": reward}) + "\n").encode("utf-8"))

    return on_run


class RolloutTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()
        self.version = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})

    def test_runs_one_per_instance_and_reads_reward(self) -> None:
        rollout = dataset_rollout(
            suite=_DemoSuite(),
            backend=FakeBackend(on_run=_simulate(0.5)),
            store=LocalDirStore(self.ws / "runs"),
            runs_root=self.ws / "runs",
        )
        slice_ = Slice("demo", ({"instance_id": "i1"}, {"instance_id": "i2"}))
        runs = rollout(self.version, slice_)
        self.assertEqual(len(runs), 2)
        self.assertEqual({r.instance_id for r in runs}, {"i1", "i2"})
        self.assertTrue(all(r.reward == 0.5 for r in runs))

    def test_reuses_existing_run_dir(self) -> None:
        backend = FakeBackend(on_run=_simulate(0.5))
        rollout = dataset_rollout(
            suite=_DemoSuite(),
            backend=backend,
            store=LocalDirStore(self.ws / "runs"),
            runs_root=self.ws / "runs",
        )
        slice_ = Slice("demo", ({"instance_id": "i1"},))
        first = rollout(self.version, slice_)
        run_id = first[0].run_id
        again = rollout(self.version, slice_)  # same (version, slice) -> reuse
        self.assertEqual(again[0].run_id, run_id)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_rollout -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent_lab.evolution.components.rollout'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/components/rollout.py
"""The default rollout: a version in, eval-suite runs out.

A thin wrapper over the existing concurrent eval driver (``run_dataset``). The
run_id is deterministic in ``(version, slice)`` so an unchanged side is measured
once and reused across steps (measurement reuse, for free). The factory binds
deployment concerns; the slice travels per call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.evals.dataset import run_dataset
from simple_agent_lab.evals.in_container import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
)
from simple_agent_lab.evals.protocols import ArtifactStore, ContainerBackend, Suite
from simple_agent_lab.evolution.types import PROVIDER_NAME, Run, Slice, Version

Rollout = Callable[[Version, Slice], Sequence[Run]]


def _provider_args(version: Version) -> tuple[str, dict[str, str]]:
    """Translate a version's provider.json into run_dataset provider kwargs."""

    path = version.dir / PROVIDER_NAME
    if not path.is_file():
        return "fake", {}
    data = json.loads(path.read_text(encoding="utf-8"))
    api = data.get("api", "fake")
    if api == "fake":
        return "fake", {}
    env = {OPENAI_MODEL_ENV: data["model"]}
    if data.get("base_url"):
        env[OPENAI_BASE_URL_ENV] = data["base_url"]
    if api in API_KIND_CHOICES:
        env[API_KIND_ENV] = api
    key_env = data.get("api_key_env", "")
    if key_env and os.environ.get(key_env):
        env[OPENAI_AUTH_ENV] = os.environ[key_env]
    return "openai", env


def dataset_rollout(
    *,
    suite: Suite,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
) -> Rollout:
    runs_root = Path(runs_root)

    def rollout(version: Version, slice_: Slice) -> Sequence[Run]:
        run_id = f"{version.hash}-{slice_.sha}"
        run_dir = runs_root / run_id
        if not _already_measured(run_dir, slice_):
            provider, provider_env = _provider_args(version)
            run_dataset(
                suite=suite,
                instances=slice_.instances,
                backend=backend,
                store=store,
                run_root=runs_root,
                run_id=run_id,
                concurrency=concurrency,
                provider=provider,
                provider_env=provider_env,
                **dict(run_kwargs or {}),
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "version.json").write_text(
                json.dumps({"version": version.hash, "slice": slice_.id, "sha": slice_.sha}),
                encoding="utf-8",
            )
        return [Run(p) for p in sorted(run_dir.iterdir()) if p.is_dir()]

    return rollout


def _already_measured(run_dir: Path, slice_: Slice) -> bool:
    if not run_dir.is_dir():
        return False
    have = {p.name for p in run_dir.iterdir() if p.is_dir()}
    want = {str(inst.get("instance_id", n)) for n, inst in enumerate(slice_.instances)}
    return want.issubset(have)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_rollout -v`
Expected: PASS (2 tests)

> If `API_KIND_CHOICES`, `API_KIND_ENV`, `OPENAI_AUTH_ENV`, `OPENAI_BASE_URL_ENV`, or `OPENAI_MODEL_ENV` are not importable from `simple_agent_lab.evals.in_container`, grep that module for the exact names (`rg "OPENAI_MODEL_ENV|API_KIND" src/simple_agent_lab/evals/in_container.py`) and fix the import — these were the names on `pr-30`.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/components/rollout.py tests/unit/test_evolution_rollout.py
git commit -m "feat(evolution): dataset_rollout over the eval harness with measurement reuse"
```

---

## Task 8: Registry + config (`registry.py`, `config.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/registry.py`
- Create: `src/simple_agent_lab/evolution/config.py`
- Test: `tests/unit/test_evolution_config.py`

The registry is a shallow `{name: factory}` dict per category. `Use(name, **args)` names a component + its kwargs. `build(category, use)` looks it up and calls it.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evolution_config.py
from __future__ import annotations

import unittest

from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.config import Use
from simple_agent_lab.evolution.components.criterion import improve


class RegistryTest(unittest.TestCase):
    def test_builtin_criterion_resolves_by_name(self) -> None:
        crit = registry.build("criterion", Use("improve", dim="reward"))
        base = {"i1": {"reward": 0.0}}
        cand = {"i1": {"reward": 1.0}}
        self.assertTrue(crit(base, cand).accepted)

    def test_register_and_build_custom(self) -> None:
        registry.REWARDS["myreward"] = lambda: (lambda run: 1.0)
        fn = registry.build("reward", Use("myreward"))
        self.assertEqual(fn(object()), 1.0)

    def test_unknown_name_lists_options(self) -> None:
        with self.assertRaises(KeyError) as cm:
            registry.build("criterion", Use("nope"))
        self.assertIn("improve", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_config -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent_lab.evolution.registry'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/config.py
"""Typed config that SELECTS components; it never hides them. Every name maps
to a factory you can grep, and ``Experiment(...)`` can bypass config entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Use:
    """A component reference: a registry name plus the kwargs its factory takes."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, **args: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", dict(args))


@dataclass(frozen=True)
class Config:
    """Names the four components + run settings. Built in Python (a YAML loader
    is a trivial later add)."""

    workspace: str | Path
    rollout: Use
    reward: Use = field(default_factory=lambda: Use("result_key"))
    strategy: Use | None = None  # Plan 2: required for agent-driven runs
    criterion: Use = field(default_factory=lambda: Use("improve", dim="reward"))
    slice_id: str = "custom"
    instances: tuple[Mapping[str, Any], ...] = ()
    auto_promote: bool = True
```

```python
# src/simple_agent_lab/evolution/registry.py
"""Shallow {name -> factory} dicts per component category. No entry-point
scanning, no metaclasses. Register a custom component by assigning into the dict
(or with the 3-line ``register`` helper below).
"""

from __future__ import annotations

from typing import Any, Callable

from simple_agent_lab.evolution.components import criterion as _criterion
from simple_agent_lab.evolution.components import reward as _reward
from simple_agent_lab.evolution.config import Use

# Each factory takes the Use.args as kwargs and returns the component callable.
ROLLOUTS: dict[str, Callable[..., Any]] = {}
REWARDS: dict[str, Callable[..., Any]] = {
    # rewards are themselves the callable; wrap so build() can pass args uniformly
    "result_key": lambda: _reward.result_key,
    "cost_tokens": lambda: _reward.cost_tokens,
}
STRATEGIES: dict[str, Callable[..., Any]] = {}
CRITERIA: dict[str, Callable[..., Any]] = {
    "improve": _criterion.improve,
    "not_worse": _criterion.not_worse,
}

_TABLES = {
    "rollout": ROLLOUTS,
    "reward": REWARDS,
    "strategy": STRATEGIES,
    "criterion": CRITERIA,
}


def register(category: str, name: str, factory: Callable[..., Any]) -> None:
    _TABLES[category][name] = factory


def build(category: str, use: Use) -> Any:
    table = _TABLES[category]
    if use.name not in table:
        raise KeyError(
            f"unknown {category} {use.name!r}; registered: {sorted(table)}"
        )
    factory = table[use.name]
    return factory(**use.args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_config -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/registry.py src/simple_agent_lab/evolution/config.py tests/unit/test_evolution_config.py
git commit -m "feat(evolution): shallow component registry + typed Config/Use"
```

---

## Task 9: Experiment + public exports (`experiment.py`, `__init__.py`)

**Files:**
- Create: `src/simple_agent_lab/evolution/experiment.py`
- Create: `src/simple_agent_lab/evolution/__init__.py`
- Test: `tests/unit/test_evolution_experiment.py`

`Experiment` only wires: it holds the four components + workspace + slice, exposes `step`/`run`/`history`/`rollback`, and seeds an initial version on first use. It accepts components directly (Level 1) or via `from_config` (Level 2).

- [ ] **Step 1: Write the failing test (end-to-end on FakeBackend)**

```python
# tests/unit/test_evolution_experiment.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import RESULT_KEY, TRACE_KEY, FakeBackend, LocalDirStore
from simple_agent_lab.evals.protocols import LaunchSpec, RunSpec
from simple_agent_lab.evolution import Experiment, Proposal
from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.components.rollout import dataset_rollout


class _DemoSuite:
    name = "demo"
    container_module = "demo.container"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="demo:latest", workdir="/work")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


def _reward_by_prompt(run_dir_to_reward):
    # FakeBackend on_run can't see the version; reward strong prompts via marker file.
    def on_run(spec: RunSpec, bound) -> None:
        bound.put(TRACE_KEY, b'{"events": []}\n')
        marker = run_dir_to_reward.get(spec.run_name, 0.3)
        bound.put(RESULT_KEY, (json.dumps({"reward": marker}) + "\n").encode("utf-8"))

    return on_run


class ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()

    def _experiment(self) -> Experiment:
        # reward depends on prompt content via a deterministic rollout stub
        def rollout(version, slice_):
            from simple_agent_lab.evolution.types import Run

            reward = 0.7 if version.read("prompt.md") == "strong" else 0.3
            run_dir = self.ws / "runs" / f"{version.hash}-{slice_.sha}" / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(json.dumps({"reward": reward}))
            return [Run(run_dir)]

        return Experiment(
            self.ws,
            rollout=rollout,
            reward=result_key,
            criterion=improve("reward"),
            slice_id="demo",
            instances=({"instance_id": "i1"},),
            seed={"prompt.md": "weak"},
        )

    def test_end_to_end_promote_history_rollback(self) -> None:
        exp = self._experiment()

        def to_strong(ctx) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, note="upgrade", kind="prompt")

        decision = exp.step(to_strong)
        self.assertTrue(decision.accepted)  # criterion 2: promotion reproduces
        self.assertEqual(exp.current().read("prompt.md"), "strong")

        # criterion 1: a second, different kind attempted and rejected
        def noop(ctx) -> Proposal:
            return Proposal(edits={"playbook.md": "x"}, note="noop", kind="playbook")

        d2 = exp.step(noop)
        self.assertFalse(d2.accepted)  # same reward -> rejected

        self.assertIn("accepted", exp.history())
        self.assertIn("rejected", exp.history())

        # criterion 3: rollback restores baseline
        exp.rollback()
        self.assertEqual(exp.current().read("prompt.md"), "weak")

        # criterion 4: rejected candidate retained in the store
        from simple_agent_lab.evolution.kernel import log

        rejected_hash = log.read(self.ws, accepted=False)[0].candidate["hash"]
        self.assertTrue((self.ws / "versions" / rejected_hash).is_dir())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.unit.test_evolution_experiment -v`
Expected: FAIL with `ImportError: cannot import name 'Experiment'`

- [ ] **Step 3: Write the implementation**

```python
# src/simple_agent_lab/evolution/experiment.py
"""Experiment: the slim wirer. Holds the four components + workspace + slice and
exposes step / run / history / rollback. No policy lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.kernel import log, loop, store
from simple_agent_lab.evolution.types import Decision, Manifest, Slice, Version


@dataclass
class _Components:
    rollout: Any
    reward: Any
    strategy: Any
    criterion: Any


class Experiment:
    """One experiment: a workspace, a way to run, a way to score, a way to judge.

    Level 1 (direct): ``Experiment(ws, rollout=fn, reward=fn, criterion=fn)``.
    Level 2 (config): ``Experiment.from_config(cfg)``.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        rollout: Any,
        reward: Any = result_key,
        criterion: Any | None = None,
        slice_id: str = "custom",
        instances: Sequence[Mapping[str, Any]] = (),
        seed: Mapping[str, str] | None = None,
        auto_promote: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self._components = _Components(
            rollout=rollout,
            reward=reward,
            strategy=None,
            criterion=criterion or improve("reward"),
        )
        self.slice = Slice(slice_id, tuple(instances))
        self.auto_promote = auto_promote
        self._ensure_seed(seed or {"prompt.md": ""})

    @classmethod
    def from_config(cls, config) -> "Experiment":
        from simple_agent_lab.evolution import registry

        exp = cls.__new__(cls)
        exp.workspace = Path(config.workspace)
        exp._components = _Components(
            rollout=registry.build("rollout", config.rollout),
            reward=registry.build("reward", config.reward),
            strategy=(
                registry.build("strategy", config.strategy)
                if config.strategy is not None
                else None
            ),
            criterion=registry.build("criterion", config.criterion),
        )
        exp.slice = Slice(config.slice_id, tuple(config.instances))
        exp.auto_promote = config.auto_promote
        exp._ensure_seed({"prompt.md": ""})
        return exp

    def _ensure_seed(self, seed: Mapping[str, str]) -> None:
        try:
            store.current(self.workspace)
        except FileNotFoundError:
            initial = store.stage(
                self.workspace,
                base=None,
                edits=dict(seed),
                manifest=Manifest(producer="experiment", note="initial"),
            )
            store.promote(self.workspace, initial)

    def current(self) -> Version:
        return store.current(self.workspace)

    def step(self, strategy: Any) -> Decision | None:
        self._components.strategy = strategy
        return loop.step(
            self.workspace, self._components, self.slice, auto_promote=self.auto_promote
        )

    def run(self, strategy: Any, *, n: int = 1) -> list[Decision]:
        self._components.strategy = strategy
        return loop.run(
            self.workspace,
            self._components,
            self.slice,
            n=n,
            auto_promote=self.auto_promote,
        )

    def history(self, *, limit: int | None = None) -> str:
        rows = log.read(self.workspace, limit=limit)
        if not rows:
            return "no decisions yet"
        return "\n".join(
            f"{d.id} [{d.kind}] {d.outcome}: {d.reason}" for d in rows
        )

    def rollback(self) -> str:
        parent = store.current(self.workspace).parent
        if not parent:
            return "already at the initial version"
        store.promote(self.workspace, store.version(self.workspace, parent))
        return f"rolled back to {parent}"
```

```python
# src/simple_agent_lab/evolution/__init__.py
"""Evolution infra: a legible, modular substrate for self-evolving agents.

User surface (plain functions, the framework owns the machinery):

    exp = Experiment(workspace, rollout=..., reward=..., criterion=...)
    def my_strategy(ctx: Context) -> Proposal | None: ...
    exp.step(my_strategy);  exp.history();  exp.rollback()

Components are swappable (rollout / reward / strategy / criterion); the kernel
(store / log / loop) owns the guarantees. Design:
docs/design/20260612-evolution-infra-redesign.md
"""

from simple_agent_lab.evolution.config import Config, Use
from simple_agent_lab.evolution.experiment import Experiment
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Proposal,
    Run,
    Slice,
    Verdict,
    Version,
)

__all__ = [
    "Experiment",
    "Config",
    "Use",
    "Context",
    "Proposal",
    "Decision",
    "Run",
    "Slice",
    "Verdict",
    "Version",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.unit.test_evolution_experiment -v`
Expected: PASS (1 test, exercising all four spec acceptance criteria)

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent_lab/evolution/experiment.py src/simple_agent_lab/evolution/__init__.py tests/unit/test_evolution_experiment.py
git commit -m "feat(evolution): Experiment wirer + public surface; e2e acceptance test"
```

---

## Task 10: Full gate (format, types, docs) and final verification

**Files:**
- Modify: `README.md` (add one line under the repo map if appropriate — optional)
- Verify: all of `src/simple_agent_lab/evolution/`

- [ ] **Step 1: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests/unit`
Expected: OK (all evolution tests + all pre-existing tests pass)

- [ ] **Step 2: Run the formatter and type checker**

Run: `uv run ruff format . && uv run ruff check src/simple_agent_lab/evolution && uv run ty check src/simple_agent_lab/evolution`
Expected: no errors. Fix any reported issues inline (common: unused imports, missing `from __future__ import annotations`).

- [ ] **Step 3: Run the docs linter**

Run: `uv run python scripts/lint_docs.py`
Expected: no errors referencing `docs/design/20260612-evolution-infra-*.md`. (Pre-existing failures in unrelated untracked skill files may be ignored.)

- [ ] **Step 4: Commit any formatting/lint fixes**

```bash
git add -A
git commit -m "chore(evolution): formatting, type, and docs-lint fixes"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Mental model (5 nouns, 4 swap points) → Task 1 (types), Tasks 4-7 (components). ✓
- Kernel (store/log/loop) → Tasks 2, 3, 6. ✓
- Scoring without measures (reward → RunScores → criterion) → Task 4 + `loop.score`/`means` (Task 6). ✓
- Config & registry (Levels 1-2, no magic) → Task 8 + `Experiment.from_config` (Task 9). ✓
- On-disk layout (versions/, pointers/, runs/, decisions.jsonl) → Tasks 2, 3, 7. ✓
- Deferred seams: `namespace` arg on store → Task 2 (`current`/`promote`); RL/selector/meta not built (correct, deferred). ✓
- MVP acceptance criteria 1-4 → Task 9 e2e test asserts all four. ✓
- Migration mapping → realized by the file structure (store←bundle, log←decisions, loop←gate, Experiment←Lab). ✓

**Deferred to Plan 2 (intentional):** LLM-agent `strategy` (`agent.py` + tools), example strategy library (`reflect_failures`, `induce_skill`), `cost_tokens` as a registry-default criterion pairing, YAML config loader. None are required for the four acceptance criteria.

**Type consistency check:** `Version`, `Run`, `Slice`, `Proposal`, `Verdict`, `Decision`, `Context`, `Manifest`, `RunScores` are defined once in `types.py` and imported everywhere; `Rollout`/`Criterion`/`RewardFn` callable aliases are consistent across `loop`, `components`, `experiment`. `store.stage(base=, edits=, manifest=)`, `log.append(...)`, and `loop.step(...)` signatures match their call sites in `loop.py` and `experiment.py`.

**Known integration risk (flagged inline in Task 7):** the `in_container` env-constant import names are taken from `pr-30`; verify against `main` and adjust if renamed.
