"""JSONL journal for dynamic workflow runs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class WorkflowJournal:
    """Append-only workflow journal plus completed-call cache.

    Rerunning a workflow script replays the same JS control flow. When an
    ``agent()`` call supplies the same stable cache key, the bridge can return a
    completed result from this journal instead of spending another subagent run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load_completed()

    def append(self, kind: str, **data: Any) -> dict[str, Any]:
        record = {"ts": time.time(), "kind": kind, **data}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            if kind == "agent_completed":
                cache_key = str(data.get("cache_key") or "")
                result = data.get("result")
                if cache_key and isinstance(result, dict):
                    self._completed[cache_key] = dict(result)
        return record

    def cached(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._completed.get(cache_key)
            return dict(cached) if cached is not None else None

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    def _load_completed(self) -> None:
        for record in self.records():
            if record.get("kind") != "agent_completed":
                continue
            cache_key = str(record.get("cache_key") or "")
            result = record.get("result")
            if cache_key and isinstance(result, dict):
                self._completed[cache_key] = dict(result)
