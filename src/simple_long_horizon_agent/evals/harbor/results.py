"""Small helpers for summarizing Harbor job outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_latest_job_dir(root: str | Path) -> Path | None:
    """Return the newest child job directory with a Harbor ``result.json``."""

    jobs_root = Path(root)
    if not jobs_root.exists():
        return None
    candidates = [
        child
        for child in jobs_root.iterdir()
        if child.is_dir() and (child / "result.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "result.json").stat().st_mtime)


def summarize_result_file(path: str | Path) -> dict[str, Any]:
    """Extract stable, dashboard-friendly fields from Harbor ``result.json``."""

    result_path = Path(path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "result_path": str(result_path),
        "job_name": data.get("job_name"),
        "n_total_trials": data.get("n_total_trials"),
        "stats": data.get("stats", {}),
    }
