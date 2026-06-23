"""Open-ended DGM driver: admit valid children from parallel branches."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.kernel.loop import means, score
from simple_agent_lab.evolution.types import Context, Manifest, Slice, Version


def run_evolution(
    workspace: Path,
    components: Any,
    slice_: Slice,
    *,
    rounds: int,
    branches: int,
    meta_workers: int | None = None,
    on_decision: Any = None,
    on_proposal_error: Any = None,
) -> list:
    tail_lock = threading.Lock()
    out: list = []
    for _ in range(max(1, rounds)):
        for decision in run_round(
            workspace,
            components,
            slice_,
            branches=branches,
            meta_workers=meta_workers,
            tail_lock=tail_lock,
            on_proposal_error=on_proposal_error,
        ):
            out.append(decision)
            if on_decision is not None:
                on_decision(decision)
    return out


def run_round(
    workspace,
    components,
    slice_,
    *,
    branches,
    meta_workers=None,
    tail_lock=None,
    on_proposal_error=None,
) -> list:
    branches = max(1, int(branches))
    meta_workers = meta_workers or branches
    tail_lock = tail_lock or threading.Lock()

    current = store.current(workspace)
    current_runs = components.rollout(current, slice_)
    runs_by_hash = {current.hash: current_runs}
    scores_by_hash = {current.hash: score(current_runs, components.reward)}

    def propose(i):
        try:
            return components.strategy(
                Context(
                    runs=tuple(current_runs),
                    current=current,
                    workspace=workspace,
                    decisions=tuple(log.read(workspace)),
                    reward=components.reward,
                )
            )
        except Exception as exc:
            if on_proposal_error is not None:
                on_proposal_error(exc)
            else:
                print(
                    f"proposal branch {i} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            return None

    with ThreadPoolExecutor(max_workers=max(1, meta_workers)) as pool:
        proposals = [p for p in pool.map(propose, range(branches)) if p is not None]
    if not proposals:
        return []

    staged_by_hash = {}
    with tail_lock:
        for proposal in proposals:
            try:
                base = (
                    store.version(workspace, proposal.base)
                    if proposal.base
                    else current
                )
            except ValueError as exc:
                if on_proposal_error is not None:
                    on_proposal_error(exc)
                continue
            if not base.dir.is_dir():
                if on_proposal_error is not None:
                    on_proposal_error(
                        ValueError(
                            f"proposal.base {proposal.base!r} is not a known version"
                        )
                    )
                continue
            cand = store.stage(
                workspace,
                base=base,
                edits=proposal.edits,
                manifest=Manifest(
                    parent=base.hash,
                    producer="dgm-open-ended",
                    evidence=proposal.evidence,
                    note=proposal.note,
                ),
            )
            staged_by_hash.setdefault(cand.hash, (proposal, base, cand))
    staged = list(staged_by_hash.values())
    if not staged:
        return []

    def roll(item):
        proposal, base, cand = item
        return proposal, base, cand, components.rollout(cand, slice_)

    with ThreadPoolExecutor(max_workers=branches) as pool:
        rolled = list(pool.map(roll, staged))

    decisions = []
    best_valid: tuple[float, Version] | None = None
    with tail_lock:
        for proposal, base, cand, cand_runs in rolled:
            if base.hash not in runs_by_hash:
                base_runs = components.rollout(base, slice_)
                runs_by_hash[base.hash] = base_runs
                scores_by_hash[base.hash] = score(base_runs, components.reward)
            base_runs = runs_by_hash[base.hash]
            base_scores = scores_by_hash[base.hash]
            cand_scores = score(cand_runs, components.reward)
            verdict = components.criterion(base_scores, cand_scores)
            valid = bool(verdict.deltas.get("valid_parent", 1.0))
            decision = log.append(
                workspace,
                baseline={"hash": base.hash, "scores": means(base_scores)},
                candidate={
                    "hash": cand.hash,
                    "parent": cand.parent,
                    "scores": means(cand_scores),
                    "note": cand.manifest.note,
                    "evidence": list(cand.manifest.evidence),
                    "valid_parent": valid,
                },
                slice_=slice_,
                verdict=verdict,
                kind=proposal.kind,
                runs={"baseline": _run_id(base_runs), "candidate": _run_id(cand_runs)},
            )
            decisions.append(decision)
            if valid:
                reward = means(cand_scores).get("reward", 0.0)
                if best_valid is None or reward > best_valid[0]:
                    best_valid = (reward, cand)
        if best_valid is not None:
            store.promote(workspace, best_valid[1])
    return decisions


def _run_id(runs) -> str:
    return runs[0].run_id if runs else ""
