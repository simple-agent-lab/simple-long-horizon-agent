"""The decision log: every gate verdict, appended forever.

This is the evolution counterpart of ``docs/decisions/`` — the repo records
its architecture decisions as ADRs, the framework records its evolution
decisions here. One JSON record per gate call (accepted, rejected, or
novelty-rejected), referencing the run directories that produced the
evidence. It is the audit trail, the rollback map, and the data behind
``hit_rate`` (the bandit prior and the meta-episode trigger).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DECISION_SCHEMA = "simple-agent-lab.decision.v1"
DECISIONS_NAME = "decisions.jsonl"

DecisionOutcome = str  # "accepted" | "rejected" | "novelty_rejected"


@dataclass(frozen=True)
class Decision:
    id: str
    level: str  # "task" | "meta"
    kind: str  # candidate kind: "lesson" | "skill" | "prompt" | ...
    decision: DecisionOutcome
    reason: str
    baseline: Mapping[str, Any]  # {"bundle": hash, "measurements": {...}}
    candidate: Mapping[str, Any]
    deltas: Mapping[str, float] = field(default_factory=dict)
    slice: Mapping[str, Any] = field(default_factory=dict)
    runs: Mapping[str, str] = field(default_factory=dict)
    episode: str = ""
    ts: str = ""
    schema: str = DECISION_SCHEMA


def _log_path(workspace: Path) -> Path:
    return workspace / DECISIONS_NAME


def next_decision_id(workspace: Path) -> str:
    return f"gate-{len(read_decisions(workspace)) + 1:06d}"


def append_decision(workspace: Path, decision: Decision) -> Decision:
    """Append one decision. The log is the single source of truth, so this is
    the only writer and it never rewrites existing lines."""

    stamped = replace(
        decision, ts=decision.ts or datetime.now(timezone.utc).isoformat()
    )
    path = _log_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(stamped)) + "\n")
    return stamped


def read_decisions(
    workspace: Path,
    *,
    level: str | None = None,
    kind: str | None = None,
    episode: str | None = None,
    limit: int | None = None,
) -> list[Decision]:
    """Read decisions oldest-first, optionally filtered; `limit` keeps the most
    recent N after filtering."""

    path = _log_path(workspace)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw.pop("schema", None)
        rows.append(Decision(**raw))
    if level is not None:
        rows = [d for d in rows if d.level == level]
    if kind is not None:
        rows = [d for d in rows if d.kind == kind]
    if episode is not None:
        rows = [d for d in rows if d.episode == episode]
    return rows[-limit:] if limit is not None else rows


def seen_candidate(workspace: Path, candidate_hash: str) -> bool:
    """Exact-duplicate novelty check: has this bundle already been judged?"""

    return any(
        d.candidate.get("bundle") == candidate_hash for d in read_decisions(workspace)
    )


def hit_rate(
    workspace: Path, *, kind: str | None = None, window: int = 20
) -> float | None:
    """Acceptance rate over the most recent gate decisions (novelty rejections
    excluded — they never reached a comparison). None when there is no data.
    This is the surface the evolution agent (and a future bandit) reads."""

    rows = [
        d
        for d in read_decisions(workspace, kind=kind, limit=None)
        if d.decision != "novelty_rejected"
    ][-window:]
    if not rows:
        return None
    return sum(1 for d in rows if d.decision == "accepted") / len(rows)
