"""Read-side views over rollout output: what ran, under which bundle, how it went.

``Run`` is the typed answer to "what is inside that path": the directory
stays the source of truth and the view only reads it. The catalog answers
the evolution agent's first question each episode — "what keeps failing,
at what cost?" — by scanning run directories on demand. Pure functions
over files; no daemon, no database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Run:
    """One instance run, as a typed read-only view over its directory.

    Layout contract (the same one ``ls`` shows):

    - ``<dir>/input/instance.json`` — what ran
    - ``<dir>/out/result.json`` — suite verdict + the standard ``reward`` key
    - ``<dir>/out/trajectory.jsonl`` — live trace: one atomically-rewritten
      snapshot record carrying ``events``

    ``dir`` is always exposed — anything not wrapped here is one ``open()``
    away. Properties are lazy: nothing is parsed until asked for.
    """

    dir: Path

    @property
    def instance_id(self) -> str:
        return self.dir.name

    @property
    def run_id(self) -> str:
        return self.dir.parent.name

    @property
    def ref(self) -> str:
        """Workspace-relative reference, ``<run_id>/<instance_id>``.

        The form durable records (decision-log evidence) must use: the
        append-only log outlives any machine, so absolute paths never
        belong in it."""

        return f"{self.run_id}/{self.instance_id}"

    @property
    def ok(self) -> bool:
        """The run produced a result (it did not crash before scoring)."""

        return (self.dir / "out" / "result.json").exists()

    @property
    def result(self) -> Mapping[str, Any]:
        """The suite verdict ({} when the run crashed)."""

        path = self.dir / "out" / "result.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @property
    def reward(self) -> float | None:
        """The standard ``result.json`` reward key (None when missing)."""

        raw = self.result.get("reward")
        return float(raw) if raw is not None else None

    @property
    def bundle(self) -> str:
        """Hash of the bundle this run was rolled out under ("" when unstamped)."""

        marker = self.dir.parent / "bundle.json"
        if not marker.exists():
            return ""
        return json.loads(marker.read_text()).get("bundle", "")

    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Trace events of the final snapshot (lazy; () when no trace)."""

        path = self.dir / "out" / "trajectory.jsonl"
        if not path.exists():
            return ()
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            return ()
        return tuple(json.loads(lines[-1]).get("events", []))


def build_catalog(runs_root: Path) -> list[Run]:
    """One ``Run`` per instance dir under ``runs_root/<run_id>/<instance_id>/``."""

    if not runs_root.exists():
        return []
    return [
        Run(instance_dir)
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir())
        for instance_dir in sorted(p for p in run_dir.iterdir() if p.is_dir())
    ]


def runs_for(runs_root: Path, *, bundle: str, instances_sha: str) -> list[Run]:
    """Instance runs of the most recent stamped run set matching
    (bundle, slice) — what lets the gate reuse a measurement instead of
    re-rolling it. Unstamped run sets never match (safe fallback: re-roll)."""

    if not runs_root.exists():
        return []
    for run_dir in sorted(
        (p for p in runs_root.iterdir() if p.is_dir()), reverse=True
    ):
        marker = run_dir / "bundle.json"
        if not marker.exists():
            continue
        stamp = json.loads(marker.read_text())
        if stamp.get("bundle") != bundle:
            continue
        if stamp.get("slice", {}).get("instances_sha") != instances_sha:
            continue
        return [Run(p) for p in sorted(run_dir.iterdir()) if p.is_dir()]
    return []


def format_rows(rows: list[Run], *, failed_only: bool = False, limit: int = 20) -> str:
    """Compact text view for the evolution agent's query_runs tool."""

    if failed_only:
        rows = [r for r in rows if r.reward is not None and r.reward <= 0.0]
    rows = rows[-limit:]
    if not rows:
        return "no runs found"
    lines = ["ref\tbundle\treward"]
    for r in rows:
        reward = "-" if r.reward is None else f"{r.reward:g}"
        lines.append(f"{r.ref}\t{r.bundle or '-'}\t{reward}")
    return "\n".join(lines)
