"""Append-only archive of evolution records, with JSONL persistence.

The archive is the experiment: every proposed candidate — accepted or not —
lands here with its payload, lineage, evaluation, and the accept/reject
reason. One JSON object per line, so a run is inspectable with `jq`/`grep`,
diff-able across methods, and resumable (`Archive.load` then keep appending).

Deliberately not a database: JSONL keeps the storage as readable as the loop
(the ADR chose legibility over ShinkaEvolve's SQLite archive).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .types import Candidate, Evaluation, EvolutionRecord


def record_to_dict(record: EvolutionRecord) -> dict[str, Any]:
    """Project a record to plain JSON-able data (payload copied to a dict)."""

    data = dataclasses.asdict(record)
    data["candidate"]["payload"] = dict(record.candidate.payload)
    data["candidate"]["parent_ids"] = list(record.candidate.parent_ids)
    data["evaluation"]["metrics"] = dict(record.evaluation.metrics)
    return data


def record_from_dict(data: Mapping[str, Any]) -> EvolutionRecord:
    """Rebuild a record from `record_to_dict` output; raises on missing keys."""

    candidate_data = data["candidate"]
    evaluation_data = data["evaluation"]
    return EvolutionRecord(
        candidate=Candidate(
            id=candidate_data["id"],
            payload=dict(candidate_data["payload"]),
            parent_ids=tuple(candidate_data["parent_ids"]),
            operator=candidate_data["operator"],
            generation=candidate_data["generation"],
            note=candidate_data.get("note", ""),
        ),
        evaluation=Evaluation(
            fitness=evaluation_data["fitness"],
            correct=evaluation_data["correct"],
            metrics=dict(evaluation_data["metrics"]),
            feedback=evaluation_data.get("feedback", ""),
            error=evaluation_data.get("error", ""),
        ),
        accepted=data["accepted"],
        reason=data["reason"],
    )


class Archive:
    """In-memory record list, optionally mirrored to a JSONL file on `add`.

    `Archive()` is ephemeral (tests, toy runs); `Archive(path=...)` persists
    every added record immediately, so a crashed or aborted run keeps all
    evaluated work; `Archive.load(path)` resumes one.
    """

    def __init__(
        self,
        records: Iterable[EvolutionRecord] = (),
        *,
        path: str | Path | None = None,
    ) -> None:
        self.records: list[EvolutionRecord] = list(records)
        self.path = Path(path) if path is not None else None

    @classmethod
    def load(cls, path: str | Path) -> Archive:
        """Load an existing archive file; subsequent `add` calls keep appending."""

        file_path = Path(path)
        records = [
            record_from_dict(json.loads(line))
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(records, path=file_path)

    def add(self, record: EvolutionRecord) -> None:
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record_to_dict(record), sort_keys=True) + "\n")

    def next_candidate_id(self) -> str:
        """Sequential ids (`c0000`, `c0001`, ...) that keep counting on resume."""

        return f"c{len(self.records):04d}"

    def population(self) -> list[EvolutionRecord]:
        """The selectable records: accepted and correct, in arrival order."""

        return [r for r in self.records if r.accepted and r.evaluation.correct]

    def best(self) -> EvolutionRecord | None:
        """Highest-fitness member of the population (earliest wins ties)."""

        population = self.population()
        if not population:
            return None
        return max(population, key=lambda r: r.evaluation.fitness)

    def top(self, k: int) -> list[EvolutionRecord]:
        """The `k` best population records, descending by fitness."""

        return sorted(
            self.population(), key=lambda r: r.evaluation.fitness, reverse=True
        )[:k]

    def __len__(self) -> int:
        return len(self.records)
