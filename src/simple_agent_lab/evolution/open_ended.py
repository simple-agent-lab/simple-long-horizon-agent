"""Open-ended DGM driver: admit valid children; select via archive."""

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
        ):
            out.append(decision)
            if on_decision is not None:
                on_decision(decision)
    return out


def run_round(
    workspace, components, slice_, *, branches, meta_workers=None, tail_lock=None
) -> list:
    branches = max(1, int(branches))
    meta_workers = meta_workers or branches
    tail_lock = tail_lock or threading.Lock()

    current = store.current(workspace)
    base_runs = components.rollout(current, slice_)
    base_scores = score(base_runs, components.reward)

    def propose(_i):
        return components.strategy(
            Context(
                runs=tuple(base_runs),
                current=current,
                workspace=workspace,
                decisions=tuple(log.read(workspace)),
                reward=components.reward,
            )
        )

    with ThreadPoolExecutor(max_workers=max(1, meta_workers)) as pool:
        proposals = [p for p in pool.map(propose, range(branches)) if p is not None]
    if not proposals:
        return []

    staged = []
    with tail_lock:
        for proposal in proposals:
            base = store.version(workspace, proposal.base) if proposal.base else current
            if not base.dir.is_dir():
                base = current
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
            staged.append((proposal, cand))

    def roll(item):
        proposal, cand = item
        return proposal, cand, components.rollout(cand, slice_)

    with ThreadPoolExecutor(max_workers=branches) as pool:
        rolled = list(pool.map(roll, staged))

    decisions = []
    best_valid: tuple[float, Version] | None = None
    with tail_lock:
        for proposal, cand, cand_runs in rolled:
            cand_scores = score(cand_runs, components.reward)
            verdict = components.criterion(base_scores, cand_scores)
            valid = bool(verdict.deltas.get("valid_parent", 1.0))
            decision = log.append(
                workspace,
                baseline={"hash": current.hash, "scores": means(base_scores)},
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
