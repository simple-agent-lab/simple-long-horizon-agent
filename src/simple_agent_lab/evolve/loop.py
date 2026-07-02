"""The evolution loop: seed, then propose -> evaluate -> decide -> record.

`run_evolution` is the whole harness. It owns only the outer structure —
budgets, lineage bookkeeping, the audit trail — and delegates every research
decision to the four seams (`propose`, `evaluate`, `select`, `accept`).
Sequential on purpose: one candidate at a time keeps the loop readable and
every archive line causally ordered. Async proposal/evaluation overlap is a
later, additive concern.

Failures are data, not crashes: a proposer or evaluator that raises produces
a rejected record carrying the error, consuming budget. A long LLM-driven
run therefore never loses evaluated work to one bad sample, and the archive
shows every dead end.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from simple_agent_lab.tools import AbortFlag

from .archive import Archive
from .select import accept_correct, select_weighted
from .types import (
    AcceptFn,
    Candidate,
    EvaluateFn,
    Evaluation,
    EvolutionRecord,
    Payload,
    ProposeFn,
    SelectFn,
)

EvolutionStatus = Literal["target_reached", "budget_exhausted", "aborted"]


def never_abort() -> bool:
    return False


@dataclass(frozen=True)
class EvolutionBudgets:
    """Stopping rules; each `None` means unbounded.

    `max_candidates` bounds the TOTAL archive size (seeds included), so
    resuming a finished run with the same budget adds nothing — the budget
    describes the experiment, not one process invocation. `target_fitness`
    stops as soon as some accepted candidate reaches it.
    """

    max_candidates: int | None = None
    wall_clock_seconds: float | None = None
    target_fitness: float | None = None


@dataclass(frozen=True)
class EvolutionResult:
    """Terminal status + the archive (the full audit trail) + counters."""

    status: EvolutionStatus
    archive: Archive
    best: EvolutionRecord | None
    candidates_added: int


def _wall_clock_abort(abort: AbortFlag, wall_clock_seconds: float | None) -> AbortFlag:
    if wall_clock_seconds is None:
        return abort
    deadline = time.monotonic() + wall_clock_seconds
    return lambda: abort() or time.monotonic() >= deadline


def _safe_evaluate(evaluate: EvaluateFn, candidate: Candidate) -> Evaluation:
    """Run the evaluator; an exception becomes an incorrect Evaluation."""

    try:
        return evaluate(candidate)
    except Exception as exc:
        return Evaluation(
            fitness=0.0, correct=False, error=f"{type(exc).__name__}: {exc}"
        )


def run_evolution(
    seeds: Sequence[Payload],
    *,
    propose: ProposeFn,
    evaluate: EvaluateFn,
    select: SelectFn | None = None,
    accept: AcceptFn = accept_correct,
    budgets: EvolutionBudgets = EvolutionBudgets(max_candidates=32),
    rng_seed: int = 0,
    archive: Archive | None = None,
    abort: AbortFlag = never_abort,
    on_record: Callable[[EvolutionRecord], None] | None = None,
) -> EvolutionResult:
    """Evolve from `seeds` until a budget, the target, or `abort` stops it.

    `seeds` are payloads evaluated first as generation 0 (skipped when
    `archive` already has records — pass `Archive.load(path)` to resume a
    persisted run, or `Archive(path=path)` to start one). `on_record` fires
    after every archived record; use it for progress printing.

    Raises `ValueError` when there is nothing to evolve from: no seeds and an
    empty archive, or every seed rejected (an accepted, correct population
    member is required before the first proposal).
    """

    archive = archive if archive is not None else Archive()
    select = select if select is not None else select_weighted()
    rng = random.Random(rng_seed)
    effective_abort = _wall_clock_abort(abort, budgets.wall_clock_seconds)
    added = 0

    def record(candidate: Candidate, evaluation: Evaluation) -> EvolutionRecord:
        decision = accept(candidate, evaluation, archive)
        entry = EvolutionRecord(
            candidate=candidate,
            evaluation=evaluation,
            accepted=decision.accepted,
            reason=decision.reason,
        )
        archive.add(entry)
        if on_record is not None:
            on_record(entry)
        return entry

    def result(status: EvolutionStatus) -> EvolutionResult:
        return EvolutionResult(
            status=status, archive=archive, best=archive.best(), candidates_added=added
        )

    def target_reached() -> bool:
        best = archive.best()
        return (
            budgets.target_fitness is not None
            and best is not None
            and best.evaluation.fitness >= budgets.target_fitness
        )

    def budget_hit() -> bool:
        return (
            budgets.max_candidates is not None
            and len(archive) >= budgets.max_candidates
        )

    if not seeds and not archive.records:
        raise ValueError("run_evolution needs seeds or a non-empty archive")

    if not archive.records:
        for payload in seeds:
            if effective_abort():
                return result("aborted")
            candidate = Candidate(
                id=archive.next_candidate_id(), payload=payload, generation=0
            )
            record(candidate, _safe_evaluate(evaluate, candidate))
            added += 1
            if target_reached():
                return result("target_reached")
            if budget_hit():
                return result("budget_exhausted")
        if not archive.population():
            raise ValueError(
                "run_evolution: every seed was rejected; nothing to select from"
            )

    while not budget_hit():
        if effective_abort():
            return result("aborted")
        if target_reached():
            return result("target_reached")

        parents = list(select(archive, rng))
        try:
            proposal = propose(parents, rng)
            candidate = Candidate(
                id=archive.next_candidate_id(),
                payload=proposal.payload,
                parent_ids=tuple(p.candidate.id for p in parents),
                operator=proposal.operator,
                generation=parents[0].candidate.generation + 1,
                note=proposal.note,
            )
            evaluation = _safe_evaluate(evaluate, candidate)
        except Exception as exc:
            # The proposer itself failed; keep the dead end in the record.
            candidate = Candidate(
                id=archive.next_candidate_id(),
                payload={},
                parent_ids=tuple(p.candidate.id for p in parents),
                operator="propose_error",
                generation=parents[0].candidate.generation + 1,
            )
            evaluation = Evaluation(
                fitness=0.0, correct=False, error=f"{type(exc).__name__}: {exc}"
            )
        record(candidate, evaluation)
        added += 1

    return result("target_reached" if target_reached() else "budget_exhausted")
