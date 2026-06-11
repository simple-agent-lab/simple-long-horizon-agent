"""The gate: measures x criteria over two rollouts, recorded as a decision.

Evaluation is deliberately generic. A ``Measure`` quantifies one named
scalar per run (reward, cost, turns, ...) plus how to aggregate; a
``Criterion`` judges the baseline's measurements against the candidate's
and produces a human-readable reason. ``gate()`` is intentionally four
steps: rollout the baseline, rollout the candidate, judge, append to the
decision log.

Measurements keep the per-instance raw values alongside the aggregate
(``MeasureFrame``): the frozen slice pairs runs by instance, so criteria
that need statistical power (``paired_improve``) read pairs instead of
comparing two means.

The rollout itself is injected (``Rollout`` is just a callable), so the
gate stays pure enough to unit-test with a stub and the same code later
runs over containerized eval suites (see evolution/rollout.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from simple_agent_lab.evolution.bundle import Bundle
from simple_agent_lab.evolution.catalog import Run, runs_for
from simple_agent_lab.evolution.decisions import (
    Decision,
    append_decision,
    next_decision_id,
    seen_comparison,
)


# --------------------------------------------------------------------------- #
# The eval slice: the frozen feedback signal of a decision.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalSlice:
    """A frozen, pinned set of instances — the feedback signal of a decision."""

    suite: str  # pinned, e.g. "swebench==0.3.1" or "demo"
    instances: tuple[Mapping[str, Any], ...]

    @property
    def instances_sha(self) -> str:
        canon = json.dumps([dict(i) for i in self.instances], sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "instances_sha": self.instances_sha,
            "n": len(self.instances),
        }


# (bundle, slice, run_id) -> the per-instance runs it produced. The slice
# arrives per call so the same rollout serves the main gate, guard slices,
# held-out rotation, and shadow evaluation. Contract: one Run per instance —
# a crashed instance has no result.json and the measures decide how to
# account for it (REWARD raises: a gate never silently compares unequal sets).
Rollout = Callable[[Bundle, EvalSlice, str], Sequence[Run]]


# --------------------------------------------------------------------------- #
# Measures: what to quantify. All read artifacts that already exist.
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class MeasureFrame:
    """One measure over one run set: the aggregate the log records, plus the
    per-instance values that produced it (the paired-statistics raw data)."""

    name: str
    value: float  # the aggregate
    per_run: Mapping[str, float]  # instance_id -> value


Measurement = Mapping[str, MeasureFrame]


@dataclass(frozen=True)
class Measure:
    """One named scalar per run, plus how to aggregate (also a policy point:
    mean for rates, sum for totals, min/median/pass@k as research demands)."""

    name: str
    per_run: Callable[[Run], float]
    aggregate: Callable[[Sequence[float]], float] = _mean

    def __call__(self, runs: Sequence[Run]) -> MeasureFrame:
        values = {run.instance_id: self.per_run(run) for run in runs}
        return MeasureFrame(self.name, self.aggregate(tuple(values.values())), values)


def _run_reward(run: Run) -> float:
    if not run.ok:
        raise FileNotFoundError(f"run {run.dir} has no out/result.json")
    if run.reward is None:
        raise ValueError(
            f"{run.dir}/out/result.json lacks the standard 'reward' key "
            "(expected a float; verifier suites map their verdict to 1.0/0.0)"
        )
    return run.reward


def _run_cost_tokens(run: Run) -> float:
    # Best-effort over the trace; a missing trace contributes zero rather
    # than failing the gate on a cost dimension.
    total = 0.0
    for event in run.events():
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            total += float(usage.get("input_tokens") or 0)
            total += float(usage.get("output_tokens") or 0)
    return total


REWARD = Measure("reward", _run_reward)
COST_TOKENS = Measure("cost_tokens", _run_cost_tokens, aggregate=sum)


# --------------------------------------------------------------------------- #
# Criteria: how to judge. Constraint-style, never weighted sums, so every
# decision carries a reason a human can read in the log.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Judgment:
    accepted: bool
    deltas: Mapping[str, float]
    reason: str


@dataclass(frozen=True)
class Criterion:
    describe: str
    judge_fn: Callable[[Measurement, Measurement], tuple[bool, str]]
    # Dimension names this criterion reads. Referencing a dimension IS
    # registering it: the Lab resolves each name to a measure ("reward" ->
    # the reward definition, built-ins like cost_tokens by name, anything
    # else -> the same-named numeric result.json field).
    requires: tuple[str, ...] = ()

    def judge(self, baseline: Measurement, candidate: Measurement) -> Judgment:
        deltas = {
            k: candidate[k].value - baseline[k].value
            for k in baseline
            if k in candidate
        }
        accepted, reason = self.judge_fn(baseline, candidate)
        return Judgment(accepted=accepted, deltas=deltas, reason=reason)


def improve(name: str, *, min_delta: float = 0.0) -> Criterion:
    """Accept when ``name`` climbs by more than ``min_delta`` (higher is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        b, c = base[name].value, cand[name].value
        ok = c - b > min_delta
        return ok, (
            f"{name} {b:.4g} -> {c:.4g} "
            f"({'meets' if ok else 'misses'} min_delta {min_delta:g})"
        )

    return Criterion(
        f"improve({name}, min_delta={min_delta:g})", judge, requires=(name,)
    )


def not_worse(name: str, *, tol: float = 0.0) -> Criterion:
    """Guard: ``name`` may not drop by more than ``tol`` (higher is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        b, c = base[name].value, cand[name].value
        ok = c >= b - tol
        return ok, f"guard {name} {b:.4g} -> {c:.4g} ({'ok' if ok else 'violated'})"

    return Criterion(f"not_worse({name}, tol={tol:g})", judge, requires=(name,))


def minimize(name: str, *, min_gain: float = 0.0) -> Criterion:
    """Accept when ``name`` shrinks by at least ``min_gain`` fraction (lower is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        b, c = base[name].value, cand[name].value
        if b == 0:
            return False, f"{name} baseline is 0; nothing to minimize"
        gain = (b - c) / b
        ok = gain >= min_gain
        return ok, f"{name} {b:.4g} -> {c:.4g} ({gain:+.1%} vs min_gain {min_gain:.0%})"

    return Criterion(
        f"minimize({name}, min_gain={min_gain:g})", judge, requires=(name,)
    )


def paired_improve(name: str, *, min_net_wins: int = 1) -> Criterion:
    """Accept when the candidate wins on more instances than it loses, by at
    least ``min_net_wins`` — sign-test style, paired per instance on the
    frozen slice. Two means a noise-width apart say little; "+9/-2 over 20
    paired instances" is evidence, and the reason in the log reads as such."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        b, c = base[name].per_run, cand[name].per_run
        paired = sorted(set(b) & set(c))
        wins = sum(1 for i in paired if c[i] > b[i])
        losses = sum(1 for i in paired if c[i] < b[i])
        ok = wins - losses >= min_net_wins
        return ok, (
            f"{name} paired over {len(paired)} instances: +{wins}/-{losses} "
            f"({'meets' if ok else 'misses'} net {min_net_wins})"
        )

    return Criterion(
        f"paired_improve({name}, net={min_net_wins})", judge, requires=(name,)
    )


def guarded(objective: Criterion, guards: Sequence[Criterion]) -> Criterion:
    """Optimize the objective subject to every guard holding."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        parts = []
        ok = True
        for guard in guards:
            g_ok, g_reason = guard.judge_fn(base, cand)
            ok = ok and g_ok
            parts.append(g_reason)
        o_ok, o_reason = objective.judge_fn(base, cand)
        parts.insert(0, o_reason)
        return ok and o_ok, "; ".join(parts)

    names = ", ".join(g.describe for g in guards)
    requires = tuple(
        dict.fromkeys(objective.requires + tuple(n for g in guards for n in g.requires))
    )
    return Criterion(f"guarded({objective.describe}, [{names}])", judge, requires)


# --------------------------------------------------------------------------- #
# The gate itself.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateResult:
    decision_id: str
    judgment: Judgment
    baseline: Measurement
    candidate: Measurement
    runs: Mapping[str, str] = field(default_factory=dict)


def _aggregates(measurement: Measurement) -> dict[str, float]:
    return {name: frame.value for name, frame in measurement.items()}


def gate(
    workspace: Path,
    *,
    baseline: Bundle,
    candidate: Bundle,
    slice_: EvalSlice,
    rollout: Rollout,
    measures: Sequence[Measure] = (REWARD,),
    criterion: Criterion | None = None,
    runs_root: Path | None = None,  # where stamped run sets live
    reuse_runs: bool = True,  # policy point: reuse (bundle, slice) measurements
    episode: str = "",
    kind: str = "",
) -> GateResult:
    """Judge ``candidate`` against ``baseline`` on a frozen slice and log it.

    Promotion is *not* done here — the gate produces evidence; the episode
    driver (or a human) moves pointers. A comparison already judged — same
    (candidate, baseline, slice) triple — is rejected before any rollout is
    spent; the same content against a *moved* baseline is novel again.

    With ``reuse_runs`` (the default), a side whose (bundle, slice) was
    already measured — stamped via ``bundle.json`` — reuses those runs
    instead of re-rolling: the unchanged baseline is measured once per
    slice, not once per candidate. The decision references the run set
    actually used, so the evidence chain stays honest; pass
    ``reuse_runs=False`` to force fresh measurements (e.g. to detect
    provider-side drift).
    """

    criterion = criterion or improve("reward")
    candidate_hash = candidate.hash
    baseline_hash = baseline.hash
    manifest = candidate.manifest
    decision_id = next_decision_id(workspace)
    runs_root = runs_root or (workspace / "runs")
    base_record: dict[str, Any] = {"bundle": baseline_hash}
    cand_record: dict[str, Any] = {
        "bundle": candidate_hash,
        "parent": manifest.parent,  # lineage, queryable without manifests
        "evidence": list(manifest.evidence),
        "note": manifest.note,
    }

    if seen_comparison(
        workspace,
        candidate=candidate_hash,
        baseline=baseline_hash,
        instances_sha=slice_.instances_sha,
    ):
        judgment = Judgment(
            False,
            {},
            f"bundle {candidate_hash} was already judged against "
            f"{baseline_hash} on this slice",
        )
        append_decision(
            workspace,
            Decision(
                id=decision_id,
                level=manifest.level,
                kind=kind or "unknown",
                decision="novelty_rejected",
                reason=judgment.reason,
                baseline=base_record,
                candidate=cand_record,
                slice=slice_.describe(),
                episode=episode,
            ),
        )
        return GateResult(decision_id, judgment, {}, {})

    def measure_side(bundle: Bundle, ref: str) -> tuple[Sequence[Run], str]:
        if reuse_runs:
            cached = runs_for(
                runs_root, bundle=bundle.hash, instances_sha=slice_.instances_sha
            )
            if cached:
                return cached, cached[0].run_id
        return rollout(bundle, slice_, ref), ref

    runs_a, base_ref = measure_side(baseline, f"{decision_id}-baseline")
    runs_b, cand_ref = measure_side(candidate, f"{decision_id}-candidate")
    base_m: Measurement = {m.name: m(runs_a) for m in measures}
    cand_m: Measurement = {m.name: m(runs_b) for m in measures}
    judgment = criterion.judge(base_m, cand_m)

    runs_ref = {"baseline": base_ref, "candidate": cand_ref}
    append_decision(
        workspace,
        Decision(
            id=decision_id,
            level=manifest.level,
            kind=kind or "unknown",
            decision="accepted" if judgment.accepted else "rejected",
            reason=f"[{criterion.describe}] {judgment.reason}",
            baseline=base_record | {"measurements": _aggregates(base_m)},
            candidate=cand_record | {"measurements": _aggregates(cand_m)},
            deltas=dict(judgment.deltas),
            slice=slice_.describe(),
            runs=runs_ref,
            episode=episode,
        ),
    )
    return GateResult(decision_id, judgment, base_m, cand_m, runs_ref)
