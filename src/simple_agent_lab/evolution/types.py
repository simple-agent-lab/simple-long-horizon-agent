"""The complete vocabulary of the evolution framework.

Two of the nouns are read-only directory views — ``Version`` and ``Run`` wrap a
directory and parse lazily, with the directory as the source of truth and
``.dir`` as the escape hatch. The rest (``Slice``, ``Proposal``, ``Verdict``,
``Decision``, ``Context``, ``Manifest``) are plain in-memory value objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.evals.instances import InstanceSet

# Aggregated/per-run score shapes shared by reward + criterion.
DimScores = Mapping[str, float]  # one run: {dim: value}
RunScores = Mapping[str, DimScores]  # {instance_id: {dim: value}}

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
        lines = [
            ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if not lines:
            return ()
        try:
            record = json.loads(lines[-1])
        except json.JSONDecodeError:
            return ()
        return tuple(record.get("events", ()))


class Slice(InstanceSet):
    """Compatibility name for the benchmark instance set used by evolution.

    New code should use ``InstanceSet``. ``Slice`` remains so existing recipes,
    tests, and logs keep working while the API migrates.
    """


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
            if isinstance(value, (int, float)):
                return float(value)
            return float(value.get("reward", 0.0))

        return tuple(r for r in self.runs if score(r) <= 0.0)

    def version(self, hash_: str) -> Version:
        return Version(self.workspace / "versions" / hash_)
