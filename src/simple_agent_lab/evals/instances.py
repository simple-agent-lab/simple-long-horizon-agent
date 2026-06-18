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
        ids = sorted(
            str(inst.get("instance_id", n)) for n, inst in enumerate(self.instances)
        )
        return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()[:12]

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
