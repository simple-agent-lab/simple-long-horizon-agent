"""Archive views and parent selection for open-ended evolution recipes.

The kernel records comparisons; this module derives a lightweight archive from
that record so recipes can branch from useful stepping stones instead of only
the current pointer.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from simple_agent_lab.evolution.kernel import log


@dataclass(frozen=True)
class ArchiveNode:
    """One selectable version in the derived evolution archive."""

    hash: str
    parent: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    decision_id: str = ""
    kind: str = ""
    valid_parent: bool = True


def nodes(workspace: Path) -> tuple[ArchiveNode, ...]:
    """Reconstruct archive nodes from the append-only decision log."""

    by_hash: dict[str, ArchiveNode] = {}
    for decision in log.read(workspace):
        baseline_hash = str(decision.baseline.get("hash", ""))
        if baseline_hash and baseline_hash not in by_hash:
            by_hash[baseline_hash] = ArchiveNode(
                hash=baseline_hash,
                scores=_scores(decision.baseline),
                decision_id=decision.id,
                kind=decision.kind,
            )

        candidate_hash = str(decision.candidate.get("hash", ""))
        if candidate_hash:
            by_hash[candidate_hash] = ArchiveNode(
                hash=candidate_hash,
                parent=_parent(decision.candidate),
                scores=_scores(decision.candidate),
                decision_id=decision.id,
                kind=decision.kind,
                valid_parent=bool(decision.candidate.get("valid_parent", True)),
            )
    return tuple(by_hash.values())


def select_parent(
    archive_nodes: Sequence[ArchiveNode],
    *,
    method: str = "best",
    dim: str = "reward",
    rng: random.Random | None = None,
) -> str:
    """Select a parent hash using common DGM/HyperAgents archive policies."""

    rng = rng or random.Random()
    candidates = [
        node for node in archive_nodes if node.valid_parent and dim in node.scores
    ]
    if not candidates:
        raise ValueError(f"no valid archive nodes with score dimension {dim!r}")

    if method == "random":
        return rng.choice(candidates).hash
    if method == "latest":
        return candidates[-1].hash
    if method == "best":
        return max(candidates, key=lambda node: node.scores[dim]).hash
    if method == "score_prop":
        return _weighted_choice(candidates, _score_weights(candidates, dim), rng)
    if method == "score_child_prop":
        score_weights = _score_weights(candidates, dim)
        child_counts = _child_counts(candidates)
        weights = [
            score_weight * math.exp(-((child_counts[node.hash] / 8) ** 3))
            for node, score_weight in zip(candidates, score_weights)
        ]
        return _weighted_choice(candidates, weights, rng)
    raise ValueError(f"unknown parent selection method {method!r}")


def _scores(record: object) -> dict[str, float]:
    if not isinstance(record, Mapping):
        return {}
    record_map = cast("Mapping[str, object]", record)
    scores = record_map.get("scores", {})
    if not isinstance(scores, Mapping):
        return {}
    return {str(dim): _float_score(value) for dim, value in scores.items()}


def _parent(record: object) -> str | None:
    if not isinstance(record, Mapping):
        return None
    record_map = cast("Mapping[str, object]", record)
    parent = record_map.get("parent")
    return str(parent) if parent else None


def _float_score(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(
            f"archive score values must be numeric, got {type(value).__name__}"
        )
    return float(value)


def _score_weights(archive_nodes: Sequence[ArchiveNode], dim: str) -> list[float]:
    scores = [node.scores[dim] for node in archive_nodes]
    top = sorted(scores, reverse=True)[:3]
    midpoint = sum(top) / len(top)
    return [1 / (1 + math.exp(-10 * (score - midpoint))) for score in scores]


def _child_counts(archive_nodes: Sequence[ArchiveNode]) -> dict[str, int]:
    counts = {node.hash: 0 for node in archive_nodes}
    for node in archive_nodes:
        if node.parent in counts:
            counts[node.parent] += 1
    return counts


def _weighted_choice(
    archive_nodes: Sequence[ArchiveNode], weights: Sequence[float], rng: random.Random
) -> str:
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(archive_nodes)).hash
    return rng.choices(list(archive_nodes), weights=weights)[0].hash
