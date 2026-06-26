"""Open-ended DGM driver: admit valid children from parallel branches."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.kernel.loop import means, score
from simple_agent_lab.evolution.progress import (
    ProgressReporter,
    mean_score,
    signed_delta,
)
from simple_agent_lab.evolution.types import Context, Manifest, Slice, Verdict, Version


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
    progress: ProgressReporter | None = None,
) -> list:
    tail_lock = threading.Lock()
    out: list = []
    total_rounds = max(1, rounds)
    for index in range(1, total_rounds + 1):
        decisions = run_round(
            workspace,
            components,
            slice_,
            branches=branches,
            meta_workers=meta_workers,
            tail_lock=tail_lock,
            on_proposal_error=on_proposal_error,
            progress=progress,
            round_index=index,
            total_rounds=total_rounds,
        )
        for decision in decisions:
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
    progress: ProgressReporter | None = None,
    round_index: int = 1,
    total_rounds: int | None = None,
) -> list:
    branches = max(1, int(branches))
    meta_workers = meta_workers or branches
    tail_lock = tail_lock or threading.Lock()
    total_rounds = total_rounds or round_index

    if progress is not None:
        progress.line(
            "dgm",
            "round",
            "start",
            index=round_index,
            total=total_rounds,
            branches=branches,
        )

    current = store.current(workspace)
    current_runs = components.rollout(current, slice_)
    runs_by_hash = {current.hash: current_runs}
    scores_by_hash = {current.hash: score(current_runs, components.reward)}

    def propose(i):
        branch = i + 1
        try:
            proposal = components.strategy(
                Context(
                    runs=tuple(current_runs),
                    current=current,
                    workspace=workspace,
                    decisions=tuple(log.read(workspace)),
                    reward=components.reward,
                )
            )
            if proposal is None:
                return None
            return branch, proposal
        except Exception as exc:
            _print_progress_proposal_failed(progress, branch, exc)
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
        _print_progress_no_proposals(progress, branches)
        _print_progress_round_complete(
            progress,
            workspace,
            round_index=round_index,
            decisions=[],
        )
        return []

    staged_by_hash = {}
    with tail_lock:
        for branch, proposal in proposals:
            try:
                base = (
                    store.version(workspace, proposal.base)
                    if proposal.base
                    else current
                )
            except ValueError as exc:
                _print_progress_proposal_failed(progress, branch, exc)
                if on_proposal_error is not None:
                    on_proposal_error(exc)
                continue
            if not base.dir.is_dir():
                exc = ValueError(
                    f"proposal.base {proposal.base!r} is not a known version"
                )
                _print_progress_proposal_failed(progress, branch, exc)
                if on_proposal_error is not None:
                    on_proposal_error(exc)
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
            if cand.hash not in staged_by_hash:
                _print_progress_candidate_staged(progress, branch, base, cand)
            staged_by_hash.setdefault(cand.hash, (branch, proposal, base, cand))
    staged = list(staged_by_hash.values())
    if not staged:
        _print_progress_no_proposals(progress, branches)
        _print_progress_round_complete(
            progress,
            workspace,
            round_index=round_index,
            decisions=[],
        )
        return []

    def roll(item):
        branch, proposal, base, cand = item
        return branch, proposal, base, cand, components.rollout(cand, slice_)

    with ThreadPoolExecutor(max_workers=branches) as pool:
        rolled = list(pool.map(roll, staged))

    decisions = []
    best_valid: tuple[float, Version] | None = None
    with tail_lock:
        for _branch, proposal, base, cand, cand_runs in rolled:
            if base.hash not in runs_by_hash:
                base_runs = components.rollout(base, slice_)
                runs_by_hash[base.hash] = base_runs
                scores_by_hash[base.hash] = score(base_runs, components.reward)
            base_runs = runs_by_hash[base.hash]
            base_scores = scores_by_hash[base.hash]
            cand_scores = score(cand_runs, components.reward)
            verdict = components.criterion(base_scores, cand_scores)
            metadata = _candidate_metadata(components, cand_runs)
            if metadata and not bool(metadata.get("valid_parent", True)):
                deltas = dict(verdict.deltas)
                deltas["valid_parent"] = 0.0
                verdict = Verdict(
                    False,
                    f"{verdict.reason}; candidate diagnostics invalid",
                    deltas,
                )
            valid = bool(verdict.deltas.get("valid_parent", 1.0))
            candidate_record = {
                "hash": cand.hash,
                "parent": cand.parent,
                "scores": means(cand_scores),
                "note": cand.manifest.note,
                "evidence": list(cand.manifest.evidence),
                "valid_parent": valid,
            }
            if metadata:
                candidate_record["diagnostics"] = metadata
            decision = log.append(
                workspace,
                baseline={"hash": base.hash, "scores": means(base_scores)},
                candidate=candidate_record,
                slice_=slice_,
                verdict=verdict,
                kind=proposal.kind,
                runs={"baseline": _run_id(base_runs), "candidate": _run_id(cand_runs)},
            )
            decisions.append(decision)
            _print_progress_decision(progress, decision, base_scores, cand_scores)
            if valid:
                reward = means(cand_scores).get("reward", 0.0)
                if best_valid is None or reward > best_valid[0]:
                    best_valid = (reward, cand)
        if best_valid is not None:
            store.promote(workspace, best_valid[1])
            if progress is not None:
                progress.line(
                    "dgm",
                    "promote",
                    version=best_valid[1].hash,
                    reward=best_valid[0],
                )
    _print_progress_round_complete(
        progress,
        workspace,
        round_index=round_index,
        decisions=decisions,
    )
    return decisions


def _run_id(runs) -> str:
    return runs[0].run_id if runs else ""


def _candidate_metadata(components: Any, runs: Any) -> dict[str, Any]:
    fn = getattr(components, "candidate_metadata", None)
    if not callable(fn):
        return {}
    metadata = fn(runs)
    return dict(metadata) if metadata else {}


def _print_progress_proposal_failed(
    progress: ProgressReporter | None, branch: int, exc: Exception
) -> None:
    if progress is None:
        return
    progress.line(
        "dgm",
        "proposal_failed",
        branch=branch,
        error=f"{type(exc).__name__}: {exc}",
    )


def _print_progress_no_proposals(
    progress: ProgressReporter | None, branches: int
) -> None:
    if progress is None:
        return
    progress.line("dgm", "no_proposals", branches=branches)


def _print_progress_candidate_staged(
    progress: ProgressReporter | None, branch: int, base: Version, cand: Version
) -> None:
    if progress is None:
        return
    progress.line(
        "dgm",
        "candidate",
        "staged",
        branch=branch,
        parent=base.hash,
        candidate=cand.hash,
    )


def _print_progress_decision(
    progress: ProgressReporter | None,
    decision: Any,
    base_scores: Any,
    cand_scores: Any,
) -> None:
    if progress is None:
        return
    baseline_reward = mean_score(base_scores)
    candidate_reward = mean_score(cand_scores)
    progress.line(
        "decision",
        decision.outcome,
        candidate=decision.candidate.get("hash"),
        baseline=decision.baseline.get("hash"),
        candidate_reward=candidate_reward,
        delta=decision.deltas.get("reward")
        if "reward" in decision.deltas
        else signed_delta(baseline_reward, candidate_reward),
        valid_parent=decision.candidate.get("valid_parent"),
        reason=decision.reason,
    )


def _print_progress_round_complete(
    progress: ProgressReporter | None,
    workspace: Path,
    *,
    round_index: int,
    decisions: list,
) -> None:
    if progress is None:
        return
    progress.line(
        "dgm",
        "round",
        "complete",
        index=round_index,
        decisions=len(decisions),
        accepted=sum(1 for decision in decisions if decision.accepted),
        current=store.current(workspace).hash,
    )
