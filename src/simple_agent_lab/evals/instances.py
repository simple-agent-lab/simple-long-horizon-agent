"""Named benchmark instance collections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InstanceSet:
    """A named, frozen set of benchmark instances for a run or comparison."""

    id: str
    instances: tuple[Mapping[str, Any], ...] = ()

    @property
    def sha(self) -> str:
        rows = []
        for n, inst in enumerate(self.instances):
            record = dict(inst)
            instance_key = str(record.get("instance_id", f"__index_{n}"))
            rows.append(
                {
                    "key": instance_key,
                    "record": record,
                }
            )
        rows.sort(key=lambda row: (row["key"], _canonical_json(row["record"])))
        payload = {"id": self.id, "instances": rows}
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:12]

    @property
    def n(self) -> int:
        return len(self.instances)


def load_jsonl_instances(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(row)
    return tuple(rows)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
