"""The decision log: one append-only JSONL record per comparison.

Kernel code. Records carry workspace-relative run refs (the log outlives any
machine). No candidate can edit this file; only the loop appends to it. The
whole file is rewritten atomically on each append (tmp-file + rename), so a
reader never observes a torn record.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.types import Decision, Slice, Verdict
from simple_agent_lab.trace.jsonl import read_jsonl, write_jsonl_atomic

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
    write_jsonl_atomic(_path(workspace), records)
    return decision


def hit_rate(
    workspace: Path, *, kind: str | None = None, window: int | None = None
) -> float | None:
    rows = read(workspace, kind=kind, limit=window)
    if not rows:
        return None
    return sum(1 for d in rows if d.accepted) / len(rows)
