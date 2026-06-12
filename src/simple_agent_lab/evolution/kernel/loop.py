"""The loop driver: observe -> propose -> compare -> record.

Kernel code. The "gate" is not a separate noun — it is this sequence (two
rollouts + apply criterion + append to the log). Promotion is host-side and
evidence-driven, so the same guarantee holds whether the strategy is a human
function or (Plan 2) an LLM agent.

``Components`` is any object exposing ``rollout``, ``reward``, ``strategy``,
``criterion`` attributes (the ``Experiment`` provides one; tests use a small
dataclass).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Manifest,
    Run,
    RunScores,
    Slice,
)


class Components(Protocol):
    rollout: Any
    reward: Any
    strategy: Any
    criterion: Any


def score(runs: Sequence[Run], reward: Any) -> dict[str, dict[str, float]]:
    """Apply ``reward`` to each run, normalizing to {instance_id: {dim: value}}."""

    out: dict[str, dict[str, float]] = {}
    for run in runs:
        value = reward(run)
        dims = value if isinstance(value, Mapping) else {"reward": float(value)}
        out[run.instance_id] = {k: float(v) for k, v in dims.items()}
    return out


def means(run_scores: RunScores) -> dict[str, float]:
    """Per-dimension mean over runs — the aggregate recorded in the log."""

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for per_dim in run_scores.values():
        for dim, val in per_dim.items():
            sums[dim] = sums.get(dim, 0.0) + val
            counts[dim] = counts.get(dim, 0) + 1
    return {dim: sums[dim] / counts[dim] for dim in sums}


def step(
    workspace: Path,
    components: Components,
    slice_: Slice,
    *,
    auto_promote: bool = True,
) -> Decision | None:
    current = store.current(workspace)
    base_runs = components.rollout(current, slice_)
    proposal = components.strategy(
        Context(
            runs=tuple(base_runs),
            current=current,
            workspace=workspace,
            decisions=tuple(log.read(workspace)),
            reward=components.reward,
        )
    )
    if proposal is None:
        return None

    base = (
        store.version(workspace, proposal.base)
        if proposal.base
        else current
    )
    candidate = store.stage(
        workspace,
        base=base,
        edits=proposal.edits,
        manifest=Manifest(
            parent=base.hash,
            producer=getattr(components.strategy, "__name__", "strategy"),
            evidence=proposal.evidence,
            note=proposal.note,
        ),
    )
    cand_runs = components.rollout(candidate, slice_)

    base_scores = score(base_runs, components.reward)
    cand_scores = score(cand_runs, components.reward)
    verdict = components.criterion(base_scores, cand_scores)

    decision = log.append(
        workspace,
        baseline={"hash": current.hash, "scores": means(base_scores)},
        candidate={
            "hash": candidate.hash,
            "parent": candidate.parent,
            "scores": means(cand_scores),
            "note": candidate.manifest.note,
            "evidence": list(candidate.manifest.evidence),
        },
        slice_=slice_,
        verdict=verdict,
        kind=proposal.kind,
        runs={
            "baseline": _run_id(base_runs),
            "candidate": _run_id(cand_runs),
        },
    )
    if verdict.accepted and auto_promote:
        store.promote(workspace, candidate)
    return decision


def run(
    workspace: Path,
    components: Components,
    slice_: Slice,
    *,
    n: int = 1,
    auto_promote: bool = True,
) -> list[Decision]:
    out = []
    for _ in range(n):
        decision = step(workspace, components, slice_, auto_promote=auto_promote)
        if decision is not None:
            out.append(decision)
    return out


def _run_id(runs: Sequence[Run]) -> str:
    return runs[0].run_id if runs else ""
