"""Read-side index over rollout output: what ran, under which bundle, how it went.

The catalog answers the evolution agent's first question each episode —
"what keeps failing, at what cost?" — by scanning run directories on demand.
Pure functions over files; no daemon, no database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogRow:
    run_id: str
    instance_id: str
    bundle: str  # bundle hash the run was rolled out under ("" when unstamped)
    reward: float | None  # None when result.json is missing or unscored
    path: str  # instance run dir, for read_trace


def build_catalog(runs_root: Path) -> list[CatalogRow]:
    """One row per instance run under ``runs_root/<run_id>/<instance_id>/``."""

    rows: list[CatalogRow] = []
    if not runs_root.exists():
        return rows
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        bundle = ""
        marker = run_dir / "bundle.json"
        if marker.exists():
            bundle = json.loads(marker.read_text()).get("bundle", "")
        for instance_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            reward: float | None = None
            result_path = instance_dir / "out" / "result.json"
            if result_path.exists():
                raw = json.loads(result_path.read_text()).get("reward")
                reward = float(raw) if raw is not None else None
            rows.append(
                CatalogRow(
                    run_id=run_dir.name,
                    instance_id=instance_dir.name,
                    bundle=bundle,
                    reward=reward,
                    path=str(instance_dir),
                )
            )
    return rows


def format_rows(
    rows: list[CatalogRow], *, failed_only: bool = False, limit: int = 20
) -> str:
    """Compact text view for the evolution agent's query_runs tool."""

    if failed_only:
        rows = [r for r in rows if r.reward is not None and r.reward <= 0.0]
    rows = rows[-limit:]
    if not rows:
        return "no runs found"
    lines = ["run_id\tinstance\tbundle\treward\tpath"]
    for r in rows:
        reward = "-" if r.reward is None else f"{r.reward:g}"
        lines.append(
            f"{r.run_id}\t{r.instance_id}\t{r.bundle or '-'}\t{reward}\t{r.path}"
        )
    return "\n".join(lines)
