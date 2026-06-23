"""Criterion combinators: how to judge candidate vs baseline.

A criterion is ``(RunScores, RunScores) -> Verdict``. Aggregation lives HERE
(each criterion receives per-run scores for both sides), so paired and
multi-dimensional judgments are possible without a separate measures system.
Declarative combinators are chosen over weighted sums because constraint-style
judgments produce auditable reasons in the decision log.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from simple_agent_lab.evolution.types import RunScores, Verdict

Criterion = Callable[[RunScores, RunScores], Verdict]


def _mean(scores: RunScores, dim: str) -> float:
    values = []
    for per_dim in scores.values():
        if dim not in per_dim:
            raise KeyError(
                f"criterion reads dimension {dim!r} but a run's scores have only "
                f"{sorted(per_dim)} — check the reward function"
            )
        values.append(per_dim[dim])
    return sum(values) / len(values) if values else 0.0


def improve(dim: str = "reward", *, min_delta: float = 0.0) -> Criterion:
    """Accept when the mean of ``dim`` strictly climbs (by at least ``min_delta``)."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        b, c = _mean(baseline, dim), _mean(candidate, dim)
        delta = c - b
        accepted = delta > 0 if min_delta == 0 else delta >= min_delta
        return Verdict(
            accepted,
            f"{dim} {b:.4g}->{c:.4g} (delta {delta:+.4g}, min_delta {min_delta})",
            {dim: delta},
        )

    return judge


def not_worse(dim: str = "reward", *, tol: float = 0.0) -> Criterion:
    """A guard: accept when ``dim`` does not drop by more than ``tol``."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        b, c = _mean(baseline, dim), _mean(candidate, dim)
        delta = c - b
        accepted = c >= b - tol
        word = "ok" if accepted else "regressed"
        return Verdict(
            accepted, f"guard {dim} {word} ({delta:+.4g}, tol {tol})", {dim: delta}
        )

    return judge


def promote_not_worse(dim: str = "reward", *, tol: float = 0.0) -> Criterion:
    """Promote when ``dim`` does not regress beyond ``tol``.

    This is the simple-recipe policy: ties are acceptable, gains are noted, and
    regressions are rejected. It differs from ``not_worse`` only in intent and
    reason wording; it is an objective policy rather than a guard combinator.
    """

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        b, c = _mean(baseline, dim), _mean(candidate, dim)
        delta = c - b
        accepted = c >= b - tol
        if delta > 0:
            word = "improved"
        elif accepted:
            word = "not worse"
        else:
            word = "regressed"
        return Verdict(
            accepted,
            f"{dim} {word} {b:.4g}->{c:.4g} (delta {delta:+.4g}, tol {tol})",
            {dim: delta},
        )

    return judge


def valid_when(dim: str = "reward") -> Criterion:
    """Open-ended admission: accept any child that produced gradable runs.

    Unlike ``improve``/``not_worse`` this never rejects on score — it admits a
    candidate (HyperAgents/DGM "valid_parent") whenever it ran and was scored,
    so worse-but-valid versions remain selectable stepping stones. The reward
    delta is still recorded for reporting; ``deltas['valid_parent']`` is the flag
    the recipe writes onto the candidate record for ``archive.select_parent``.
    """

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        valid = len(candidate) > 0
        delta = (
            _mean(candidate, dim) - _mean(baseline, dim) if valid and baseline else 0.0
        )
        word = "valid" if valid else "invalid"
        return Verdict(
            valid,
            f"admission {word} (reward {delta:+.4g})",
            {dim: delta, "valid_parent": 1.0 if valid else 0.0},
        )

    return judge


def guarded(objective: Criterion, guards: Sequence[Criterion]) -> Criterion:
    """Optimize ``objective`` subject to every guard holding ("X subject to Y")."""

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        obj = objective(baseline, candidate)
        deltas = dict(obj.deltas)
        reasons = [obj.reason]
        accepted = obj.accepted
        for guard in guards:
            g = guard(baseline, candidate)
            deltas.update(g.deltas)
            reasons.append(g.reason)
            accepted = accepted and g.accepted
        return Verdict(accepted, "; ".join(reasons), deltas)

    return judge
