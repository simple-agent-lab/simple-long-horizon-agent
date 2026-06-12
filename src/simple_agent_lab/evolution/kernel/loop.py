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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Manifest,
    Proposal,
    RewardFn,
    Run,
    RunScores,
    Slice,
    Verdict,
    Version,
)


class Components(Protocol):
    rollout: Callable[[Version, Slice], Sequence[Run]]
    reward: RewardFn
    strategy: Callable[[Context], Proposal | None]
    criterion: Callable[[RunScores, RunScores], Verdict]


def score(runs: Sequence[Run], reward: RewardFn) -> dict[str, dict[str, float]]:
    """Apply ``reward`` to each run, normalizing to {instance_id: {dim: value}}."""

    out: dict[str, dict[str, float]] = {}
    for run in runs:
        value = reward(run)
        if isinstance(value, Mapping):
            mapping = cast("Mapping[str, float]", value)
            dims = {str(k): float(v) for k, v in mapping.items()}
        else:
            dims = {"reward": float(value)}
        out[run.instance_id] = dims
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
    """Run one observe -> propose -> compare -> record cycle.

    Returns the logged ``Decision``, or ``None`` if the strategy declined.
    """

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

    if proposal.base:
        base = store.version(workspace, proposal.base)
        if not base.dir.is_dir():
            raise ValueError(
                f"proposal.base {proposal.base!r} is not a known version"
            )
    else:
        base = current
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
    _check_pair(base_runs, cand_runs)

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
    # log first, then promote: prefer an accepted-but-unpromoted record over a promoted version with no log entry
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
    """Call ``step`` up to ``n`` times, collecting the non-declined decisions."""

    out = []
    for _ in range(n):
        decision = step(workspace, components, slice_, auto_promote=auto_promote)
        if decision is not None:
            out.append(decision)
    return out


def _check_pair(base_runs: Sequence[Run], cand_runs: Sequence[Run]) -> None:
    base_ids = {r.instance_id for r in base_runs}
    cand_ids = {r.instance_id for r in cand_runs}
    if not base_ids or not cand_ids:
        raise ValueError("rollout produced no runs; cannot compare")
    if base_ids != cand_ids:
        raise ValueError(
            f"baseline/candidate ran different instances: "
            f"{sorted(base_ids)} vs {sorted(cand_ids)}"
        )


def _run_id(runs: Sequence[Run]) -> str:
    return runs[0].run_id if runs else ""
