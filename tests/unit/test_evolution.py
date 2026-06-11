"""Behavioral tests for the evolution substrate (bundle / decisions / gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_agent_lab.evolution import (
    Bundle,
    EvalSlice,
    Lab,
    Manifest,
    MeasureFrame,
    Run,
    build_catalog,
    gate,
    guarded,
    hit_rate,
    improve,
    minimize,
    not_worse,
    paired_improve,
    promote,
    read_decisions,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.agent import EvolutionConfig, make_evolution_tools
from simple_agent_lab.tools import tool_result_text


def make_initial(workspace: Path, *, prompt: str = "be helpful") -> Bundle:
    bundle = stage_bundle(
        workspace,
        manifest=Manifest(level="task", producer="test"),
        edits={"prompt.md": prompt},
    )
    promote(workspace, "task", bundle)
    return bundle


def stub_rollout(rewards_by_prompt: dict[str, float], runs_root: Path):
    """A Rollout whose reward depends on the bundle's prompt content."""

    def rollout(bundle: Bundle, slice_: EvalSlice, run_id: str) -> list[Run]:
        reward = rewards_by_prompt[bundle.read("prompt.md")]
        instance = runs_root / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [Run(instance)]

    return rollout


def frames(**values: float) -> dict[str, MeasureFrame]:
    """Aggregate-only measurements for criteria tests."""

    return {name: MeasureFrame(name, value, {}) for name, value in values.items()}


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def test_bundle_hash_ignores_manifest_and_is_content_addressed(tmp_path):
    a = stage_bundle(
        tmp_path, manifest=Manifest(level="task", note="one"), edits={"prompt.md": "x"}
    )
    b = stage_bundle(
        tmp_path, manifest=Manifest(level="task", note="two"), edits={"prompt.md": "x"}
    )
    assert a == b  # same content -> same directory, manifest notes don't matter
    # Re-staging identical content must NOT rewrite the archived lineage:
    # first provenance wins (rollback walks manifest parents).
    assert a.manifest.note == "one"
    c = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), edits={"prompt.md": "y"}
    )
    assert c.hash != a.hash


def test_stage_rejects_escaping_paths(tmp_path):
    with pytest.raises(ValueError):
        stage_bundle(
            tmp_path, manifest=Manifest(level="task"), edits={"../escape.md": "x"}
        )


def test_promote_resolve_and_rollback(tmp_path):
    first = make_initial(tmp_path, prompt="v1")
    second = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), edits={"prompt.md": "v2"}, base=first
    )
    promote(tmp_path, "task", second)
    assert resolve(tmp_path, "task") == second
    promote(tmp_path, "task", first)  # rollback = move the pointer back
    assert resolve(tmp_path, "task") == first
    assert second.dir.exists()  # nothing is deleted


def test_stage_from_base_inherits_files(tmp_path):
    base = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task"),
        edits={"prompt.md": "p", "lessons.jsonl": "l1"},
    )
    child = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task", parent=base.hash),
        base=base,
        edits={"lessons.jsonl": "l1\nl2"},
    )
    assert child.read("prompt.md") == "p"
    assert child.parent == base.hash


def test_stage_edits_support_tombstones_and_bytes(tmp_path):
    base = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task"),
        edits={"prompt.md": "p", "skills/old/SKILL.md": "obsolete"},
    )
    child = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task", parent=base.hash),
        base=base,
        edits={"skills/old/SKILL.md": None, "assets/logo.png": b"\x89PNG"},
    )
    assert child.read("prompt.md") == "p"
    assert "skills/old/SKILL.md" not in child.files()  # tombstone retired it
    assert (child.dir / "assets/logo.png").read_bytes() == b"\x89PNG"


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def test_criteria_judgments():
    base = frames(reward=0.4, cost_tokens=100.0)
    better = frames(reward=0.6, cost_tokens=100.0)
    cheaper = frames(reward=0.4, cost_tokens=80.0)
    assert improve("reward").judge(base, better).accepted
    assert not improve("reward").judge(base, base).accepted  # needs strict climb
    assert minimize("cost_tokens", min_gain=0.1).judge(base, cheaper).accepted
    efficiency = guarded(minimize("cost_tokens", min_gain=0.1), [not_worse("reward")])
    verdict = efficiency.judge(base, cheaper)
    assert verdict.accepted and "guard reward" in verdict.reason
    worse = frames(reward=0.2, cost_tokens=50.0)
    assert not efficiency.judge(base, worse).accepted  # guard fires


def test_paired_criterion_reads_per_instance_values():
    # Means a hair apart say little; per-instance pairing is the evidence.
    base = {"reward": MeasureFrame("reward", 0.33, {"i1": 1.0, "i2": 0.0, "i3": 0.0})}
    cand = {"reward": MeasureFrame("reward", 0.66, {"i1": 1.0, "i2": 1.0, "i3": 0.0})}
    verdict = paired_improve("reward").judge(base, cand)
    assert verdict.accepted and "+1/-0" in verdict.reason
    lateral = {"reward": MeasureFrame("reward", 0.33, {"i1": 0.0, "i2": 1.0, "i3": 0.0})}
    assert not paired_improve("reward").judge(base, lateral).accepted  # +1/-1 nets 0


# --------------------------------------------------------------------------- #
# gate + decision log
# --------------------------------------------------------------------------- #
def test_gate_accepts_logs_and_novelty_rejects_duplicates(tmp_path):
    baseline = make_initial(tmp_path, prompt="weak")
    candidate = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task", note="try strong"),
        base=baseline,
        edits={"prompt.md": "strong"},
    )
    rollout = stub_rollout({"weak": 0.3, "strong": 0.7}, tmp_path / "runs")
    slice_ = EvalSlice(suite="demo", instances=({"instance_id": "i1"},))

    result = gate(
        tmp_path,
        baseline=baseline,
        candidate=candidate,
        slice_=slice_,
        rollout=rollout,
        kind="prompt",
    )
    assert result.judgment.accepted
    assert result.candidate["reward"].value == 0.7
    assert result.candidate["reward"].per_run == {"i1": 0.7}  # paired raw data

    decisions = read_decisions(tmp_path)
    assert [d.decision for d in decisions] == ["accepted"]
    assert decisions[0].slice["n"] == 1 and decisions[0].kind == "prompt"

    # The same candidate again costs zero rollouts and is logged as such.
    again = gate(
        tmp_path,
        baseline=baseline,
        candidate=candidate,
        slice_=slice_,
        rollout=rollout,
        kind="prompt",
    )
    assert not again.judgment.accepted
    assert read_decisions(tmp_path)[-1].decision == "novelty_rejected"
    assert hit_rate(tmp_path) == 1.0  # novelty rejections don't dilute the rate


def test_novelty_is_keyed_on_the_comparison_not_the_content(tmp_path):
    baseline = make_initial(tmp_path, prompt="weak")
    candidate = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task"),
        base=baseline,
        edits={"prompt.md": "strong"},
    )
    rollout = stub_rollout(
        {"weak": 0.3, "mid": 0.5, "strong": 0.7}, tmp_path / "runs"
    )
    slice_ = EvalSlice(suite="demo", instances=())
    gate(
        tmp_path,
        baseline=baseline,
        candidate=candidate,
        slice_=slice_,
        rollout=rollout,
        kind="prompt",
    )
    # Same content, but the baseline moved: a fresh comparison, not a dup —
    # archived stepping stones stay re-testable.
    moved = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), edits={"prompt.md": "mid"}
    )
    result = gate(
        tmp_path,
        baseline=moved,
        candidate=candidate,
        slice_=slice_,
        rollout=rollout,
        kind="prompt",
    )
    assert result.judgment.accepted
    assert read_decisions(tmp_path)[-1].decision == "accepted"


def test_gate_reuses_stamped_measurements(tmp_path):
    rewards = {"weak": 0.3, "strong": 0.7, "stronger": 0.9}
    rolled = []

    def rollout(bundle, slice_, run_id):
        rolled.append(bundle.read("prompt.md"))
        run_dir = tmp_path / "runs" / run_id
        instance = run_dir / "i1"
        (instance / "out").mkdir(parents=True)
        (instance / "out" / "result.json").write_text(
            json.dumps({"reward": rewards[bundle.read("prompt.md")]})
        )
        run_dir.joinpath("bundle.json").write_text(
            json.dumps({"bundle": bundle.hash, "slice": slice_.describe()})
        )
        return [Run(instance)]

    baseline = make_initial(tmp_path, prompt="weak")
    slice_ = EvalSlice(suite="demo", instances=())
    first = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), base=baseline,
        edits={"prompt.md": "strong"},
    )
    gate(tmp_path, baseline=baseline, candidate=first, slice_=slice_,
         rollout=rollout, kind="prompt")
    assert rolled == ["weak", "strong"]

    # Second gate against the same baseline+slice: the baseline measurement
    # is reused from its stamp — only the new candidate rolls out.
    second = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), base=baseline,
        edits={"prompt.md": "stronger"},
    )
    result = gate(tmp_path, baseline=baseline, candidate=second, slice_=slice_,
                  rollout=rollout, kind="prompt")
    assert rolled == ["weak", "strong", "stronger"]  # no second "weak"
    assert result.judgment.accepted
    decision = read_decisions(tmp_path)[-1]
    assert decision.runs["baseline"] == "gate-000001-baseline"  # honest reference


def test_gate_rejects_when_candidate_does_not_improve(tmp_path):
    baseline = make_initial(tmp_path, prompt="ok")
    candidate = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task"),
        base=baseline,
        edits={"prompt.md": "worse"},
    )
    rollout = stub_rollout({"ok": 0.5, "worse": 0.4}, tmp_path / "runs")
    result = gate(
        tmp_path,
        baseline=baseline,
        candidate=candidate,
        slice_=EvalSlice(suite="demo", instances=()),
        rollout=rollout,
        kind="prompt",
    )
    assert not result.judgment.accepted
    assert read_decisions(tmp_path)[-1].deltas["reward"] == pytest.approx(-0.1)


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
def test_catalog_indexes_runs_with_bundle_stamp(tmp_path):
    runs = tmp_path / "runs"
    instance = runs / "r1" / "i1" / "out"
    instance.mkdir(parents=True)
    (instance / "result.json").write_text(json.dumps({"reward": 0.0}))
    (runs / "r1" / "bundle.json").write_text(json.dumps({"bundle": "abc123"}))
    rows = build_catalog(runs)
    assert len(rows) == 1
    assert rows[0].bundle == "abc123" and rows[0].reward == 0.0


# --------------------------------------------------------------------------- #
# Lab: the user surface (strategy + reward as plain functions)
# --------------------------------------------------------------------------- #
def test_lab_step_with_custom_strategy_and_reward(tmp_path):
    def rollout(bundle, slice_, run_id):
        # Score lives in a custom result field; the reward fn below reads it.
        strength = "strong" in bundle.read("prompt.md")
        instance = tmp_path / "runs" / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        (instance / "out" / "result.json").write_text(
            json.dumps({"resolved": strength})
        )
        return [Run(instance)]

    def my_reward(run):  # verl-style: one typed Run in, one float out
        return 1.0 if run.result["resolved"] else 0.0

    def my_strategy(ctx):
        assert ctx.failures  # the observe rollout of the weak baseline
        return ctx.propose(
            kind="prompt",
            edits={"prompt.md": ctx.current.read("prompt.md") + " strong"},
            note="strengthen prompt",
            evidence=[ctx.failures[0].ref],  # the convention: refs, not paths
        )

    lab = Lab(tmp_path, rollout=rollout, reward=my_reward, seed={"prompt.md": "weak"})
    report = lab.step(my_strategy)
    assert report.accepted and report.promoted_to
    assert "strong" in resolve(tmp_path, "task").read("prompt.md")
    assert "ACCEPTED" in report.text and "accepted" in lab.history()
    # Evidence lands in the append-only log as a workspace-relative ref.
    assert read_decisions(tmp_path)[-1].candidate["evidence"] == [
        "step-000001-observe/i1"
    ]

    # A strategy may also decline to propose.
    assert not lab.step(lambda ctx: None).proposed


def test_lab_branches_from_a_rejected_stepping_stone(tmp_path):
    # "risky" regresses on its own, but is the right base for the real fix —
    # the DGM stepping-stone move: branch from an archived (rejected) bundle.
    rewards = {"v1": 0.5, "risky": 0.4, "risky tuned": 0.9}
    lab = Lab(
        tmp_path,
        rollout=stub_rollout(rewards, tmp_path / "runs"),
        seed={"prompt.md": "v1"},
    )
    first = lab.step(
        lambda ctx: ctx.propose(kind="prompt", edits={"prompt.md": "risky"}, note="")
    )
    assert not first.accepted and first.candidate

    def branch_from_rejected(ctx):
        rejected = next(d for d in ctx.decisions if d.decision == "rejected")
        base = rejected.candidate["bundle"]
        return ctx.propose(
            kind="prompt",
            edits={"prompt.md": ctx.bundle(base).read("prompt.md") + " tuned"},
            note="tune the rejected variant",
            base=base,
        )

    second = lab.step(branch_from_rejected)
    assert second.accepted and second.promoted_to
    promoted = resolve(tmp_path, "task")
    assert promoted.read("prompt.md") == "risky tuned"
    assert promoted.parent == first.candidate  # lineage = the tree
    # The decision log carries the lineage too — tree analytics (child
    # counts) never need to scan manifests.
    assert read_decisions(tmp_path)[-1].candidate["parent"] == first.candidate

    unknown = lab.step(
        lambda ctx: ctx.propose(
            kind="prompt", edits={"prompt.md": "x"}, note="", base="nosuchhash"
        )
    )
    assert not unknown.accepted and "unknown base" in unknown.text


def test_lab_manual_promotion_tier(tmp_path):
    rewards = {"v1": 0.4, "v2": 0.8}
    lab = Lab(
        tmp_path,
        rollout=stub_rollout(rewards, tmp_path / "runs"),
        seed={"prompt.md": "v1"},
        auto_promote=False,
    )
    report = lab.step(
        lambda ctx: ctx.propose(kind="prompt", edits={"prompt.md": "v2"}, note="")
    )
    assert report.accepted and not report.promoted_to  # gated, awaiting a human
    assert "v1" in resolve(tmp_path, "task").read("prompt.md")

    assert "refused" in lab.promote("deadbeefdead")  # no evidence, no pointer move
    assert "promoted" in lab.promote(report.candidate)
    assert "v2" in resolve(tmp_path, "task").read("prompt.md")


def test_lab_proposal_carries_its_own_criterion(tmp_path):
    rewards = {"v1": 0.5, "v2": 0.5}  # equal reward: improve() would reject
    lab = Lab(
        tmp_path,
        rollout=stub_rollout(rewards, tmp_path / "runs"),
        seed={"prompt.md": "v1"},
    )
    report = lab.step(
        lambda ctx: ctx.propose(
            kind="prompt",
            edits={"prompt.md": "v2"},
            note="lateral move",
            criterion=not_worse("reward"),
        )
    )
    assert report.accepted
    assert "not_worse" in read_decisions(tmp_path)[-1].reason


def test_lab_criterion_dimensions_resolve_from_result_fields(tmp_path):
    # Referencing a dimension in the criterion IS registering it: unknown
    # names resolve to the same-named result.json field — no Measure
    # plumbing in user code.
    def rollout(bundle, slice_, run_id):
        instance = tmp_path / "runs" / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        reward = 0.6 if "v2" in bundle.read("prompt.md") else 0.4
        (instance / "out" / "result.json").write_text(
            json.dumps({"reward": reward, "compile_ok": 1.0})
        )
        return [Run(instance)]

    lab = Lab(
        tmp_path,
        rollout=rollout,
        seed={"prompt.md": "v1"},
        criterion=guarded(improve("reward"), [not_worse("compile_ok")]),
    )
    report = lab.step(
        lambda ctx: ctx.propose(kind="prompt", edits={"prompt.md": "v2"}, note="")
    )
    assert report.accepted and "guard compile_ok" in report.text


def test_lab_rollback_returns_to_parent(tmp_path):
    def rollout(bundle, slice_, run_id):
        instance = tmp_path / "runs" / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        reward = 1.0 if "v2" in bundle.read("prompt.md") else 0.0
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [Run(instance)]

    lab = Lab(tmp_path, rollout=rollout, seed={"prompt.md": "v1"})
    lab.step(
        lambda ctx: ctx.propose(kind="prompt", edits={"prompt.md": "v2"}, note="v2")
    )
    assert "v2" in resolve(tmp_path, "task").read("prompt.md")
    assert "rolled back" in lab.rollback()
    assert "v1" in resolve(tmp_path, "task").read("prompt.md")
    assert "already at the initial version" in lab.rollback()


# --------------------------------------------------------------------------- #
# evolution agent tools (called directly; no model in the loop)
# --------------------------------------------------------------------------- #
def test_tools_stage_and_gate_a_candidate(tmp_path):
    baseline = make_initial(tmp_path, prompt="weak")
    promote(tmp_path, "meta", baseline)  # any meta bundle; prompt unused here
    rollout = stub_rollout({"weak": 0.3, "strong": 0.7}, tmp_path / "runs")
    config = EvolutionConfig(
        workspace=tmp_path,
        rollout=rollout,
        slice_=EvalSlice(suite="demo", instances=()),
        max_gates_per_episode=1,
    )
    tools = {t.name: t for t in make_evolution_tools(config, episode="ep-1")}

    staged = tools["write_candidate"].execute(
        "c1",
        {
            "kind": "prompt",
            "edits": {"prompt.md": "strong"},
            "note": "try strong",
            "evidence": ["trace:t1"],
        },
        lambda: False,
        None,
    )
    candidate_hash = tool_result_text(staged).split()[2]

    gated = tools["run_gate"].execute(
        "c2", {"candidate": candidate_hash, "kind": "prompt"}, lambda: False, None
    )
    assert "ACCEPTED" in tool_result_text(gated)

    # Budget: a second gate call in the same episode is refused.
    refused = tools["run_gate"].execute(
        "c3", {"candidate": candidate_hash, "kind": "prompt"}, lambda: False, None
    )
    assert refused.is_error

    log_text = tool_result_text(
        tools["read_decisions"].execute("c4", {}, lambda: False, None)
    )
    assert "accepted" in log_text
