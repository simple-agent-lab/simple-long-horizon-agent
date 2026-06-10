"""The gate: measures x criteria over two rollouts, recorded as a decision.

Evaluation is deliberately generic. A ``Measure`` quantifies one named scalar
over a set of run directories (reward, cost, turns, ...); a ``Criterion``
judges a baseline measurement set against a candidate's and produces a
human-readable reason. ``gate()`` is intentionally four steps: rollout the
baseline, rollout the candidate, judge, append to the decision log.

The rollout itself is injected (``Rollout`` is just a callable), so the gate
stays pure enough to unit-test with a stub and the same code later runs over
containerized eval suites (see evolution/rollout.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from simple_agent_lab.evolution.bundle import bundle_hash, read_manifest
from simple_agent_lab.evolution.decisions import (
    Decision,
    append_decision,
    next_decision_id,
    seen_candidate,
)


Measurement = dict[str, float]

# (bundle_dir, run_id) -> the per-instance run directories it produced.
# Each run dir follows the eval layout: out/result.json (+ out/trajectory.jsonl).
Rollout = Callable[[Path, str], Sequence[Path]]


# --------------------------------------------------------------------------- #
# Measures: what to quantify. All read artifacts that already exist.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Measure:
    name: str
    fn: Callable[[Sequence[Path]], float]

    def __call__(self, runs: Sequence[Path]) -> float:
        return self.fn(runs)


def _result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "out" / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"run {run_dir} has no out/result.json")
    return json.loads(path.read_text())


def _mean_reward(runs: Sequence[Path]) -> float:
    rewards = []
    for run_dir in runs:
        result = _result(run_dir)
        if "reward" not in result:
            raise ValueError(
                f"{run_dir}/out/result.json lacks the standard 'reward' key "
                "(expected a float; verifier suites map their verdict to 1.0/0.0)"
            )
        rewards.append(float(result["reward"]))
    return sum(rewards) / len(rewards) if rewards else 0.0


def _total_cost_tokens(runs: Sequence[Path]) -> float:
    # Best-effort sum over trace events carrying token usage; a missing trace
    # contributes zero rather than failing the gate on a cost dimension.
    total = 0.0
    for run_dir in runs:
        trace_path = run_dir / "out" / "trajectory.jsonl"
        if not trace_path.exists():
            continue
        for line in trace_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for event in record.get("events", []):
                usage = event.get("usage") or {}
                if isinstance(usage, dict):
                    total += float(usage.get("input_tokens") or 0)
                    total += float(usage.get("output_tokens") or 0)
    return total


REWARD = Measure("reward", _mean_reward)
COST_TOKENS = Measure("cost_tokens", _total_cost_tokens)


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

    def judge(self, baseline: Measurement, candidate: Measurement) -> Judgment:
        deltas = {k: candidate[k] - baseline[k] for k in baseline if k in candidate}
        accepted, reason = self.judge_fn(baseline, candidate)
        return Judgment(accepted=accepted, deltas=deltas, reason=reason)


def improve(name: str, *, min_delta: float = 0.0) -> Criterion:
    """Accept when ``name`` climbs by more than ``min_delta`` (higher is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        delta = cand[name] - base[name]
        ok = delta > min_delta
        return ok, (
            f"{name} {base[name]:.4g} -> {cand[name]:.4g} "
            f"({'meets' if ok else 'misses'} min_delta {min_delta:g})"
        )

    return Criterion(f"improve({name}, min_delta={min_delta:g})", judge)


def not_worse(name: str, *, tol: float = 0.0) -> Criterion:
    """Guard: ``name`` may not drop by more than ``tol`` (higher is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        ok = cand[name] >= base[name] - tol
        return (
            ok,
            f"guard {name} {base[name]:.4g} -> {cand[name]:.4g} ({'ok' if ok else 'violated'})",
        )

    return Criterion(f"not_worse({name}, tol={tol:g})", judge)


def minimize(name: str, *, min_gain: float = 0.0) -> Criterion:
    """Accept when ``name`` shrinks by at least ``min_gain`` fraction (lower is better)."""

    def judge(base: Measurement, cand: Measurement) -> tuple[bool, str]:
        if base[name] == 0:
            return False, f"{name} baseline is 0; nothing to minimize"
        gain = (base[name] - cand[name]) / base[name]
        ok = gain >= min_gain
        return (
            ok,
            f"{name} {base[name]:.4g} -> {cand[name]:.4g} ({gain:+.1%} vs min_gain {min_gain:.0%})",
        )

    return Criterion(f"minimize({name}, min_gain={min_gain:g})", judge)


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
    return Criterion(f"guarded({objective.describe}, [{names}])", judge)


# --------------------------------------------------------------------------- #
# The gate itself.
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


@dataclass(frozen=True)
class GateResult:
    decision_id: str
    judgment: Judgment
    baseline: Measurement
    candidate: Measurement
    runs: Mapping[str, str] = field(default_factory=dict)


def gate(
    workspace: Path,
    *,
    baseline: Path,
    candidate: Path,
    slice_: EvalSlice,
    rollout: Rollout,
    measures: Sequence[Measure] = (REWARD,),
    criterion: Criterion | None = None,
    episode: str = "",
    kind: str = "",
) -> GateResult:
    """Judge ``candidate`` against ``baseline`` on a frozen slice and log it.

    Promotion is *not* done here — the gate produces evidence; the episode
    driver (or a human) moves pointers. Exact-duplicate candidates are
    rejected before any rollout is spent.
    """

    criterion = criterion or improve("reward")
    candidate_hash = bundle_hash(candidate)
    baseline_hash = bundle_hash(baseline)
    manifest = read_manifest(candidate)
    decision_id = next_decision_id(workspace)
    base_record: dict[str, Any] = {"bundle": baseline_hash}
    cand_record: dict[str, Any] = {
        "bundle": candidate_hash,
        "evidence": list(manifest.evidence),
        "note": manifest.note,
    }

    if seen_candidate(workspace, candidate_hash):
        judgment = Judgment(False, {}, f"bundle {candidate_hash} was already judged")
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

    runs_a = rollout(baseline, f"{decision_id}-baseline")
    runs_b = rollout(candidate, f"{decision_id}-candidate")
    base_m: Measurement = {m.name: m(runs_a) for m in measures}
    cand_m: Measurement = {m.name: m(runs_b) for m in measures}
    judgment = criterion.judge(base_m, cand_m)

    runs_ref = {
        "baseline": f"{decision_id}-baseline",
        "candidate": f"{decision_id}-candidate",
    }
    append_decision(
        workspace,
        Decision(
            id=decision_id,
            level=manifest.level,
            kind=kind or "unknown",
            decision="accepted" if judgment.accepted else "rejected",
            reason=f"[{criterion.describe}] {judgment.reason}",
            baseline=base_record | {"measurements": base_m},
            candidate=cand_record | {"measurements": cand_m},
            deltas=dict(judgment.deltas),
            slice=slice_.describe(),
            runs=runs_ref,
            episode=episode,
        ),
    )
    return GateResult(decision_id, judgment, base_m, cand_m, runs_ref)
