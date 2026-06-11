"""Behavioral tests for the evolution substrate (bundle / decisions / gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_agent_lab.evolution import (
    EvalSlice,
    Lab,
    Manifest,
    bundle_hash,
    build_catalog,
    gate,
    guarded,
    hit_rate,
    improve,
    minimize,
    not_worse,
    promote,
    read_decisions,
    read_manifest,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.agent import EvolutionConfig, make_evolution_tools
from simple_agent_lab.tools import tool_result_text


def make_initial(workspace: Path, *, prompt: str = "be helpful") -> Path:
    bundle = stage_bundle(
        workspace,
        manifest=Manifest(level="task", producer="test"),
        edits={"prompt.md": prompt},
    )
    promote(workspace, "task", bundle)
    return bundle


def stub_rollout(rewards_by_prompt: dict[str, float], runs_root: Path):
    """A Rollout whose reward depends on the bundle's prompt content."""

    def rollout(bundle_dir: Path, run_id: str) -> list[Path]:
        reward = rewards_by_prompt[(bundle_dir / "prompt.md").read_text()]
        instance = runs_root / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [instance]

    return rollout


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
    c = stage_bundle(
        tmp_path, manifest=Manifest(level="task"), edits={"prompt.md": "y"}
    )
    assert bundle_hash(c) != bundle_hash(a)


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
    assert second.exists()  # nothing is deleted


def test_stage_from_base_inherits_files(tmp_path):
    base = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task"),
        edits={"prompt.md": "p", "lessons.jsonl": "l1"},
    )
    child = stage_bundle(
        tmp_path,
        manifest=Manifest(level="task", parent=bundle_hash(base)),
        base=base,
        edits={"lessons.jsonl": "l1\nl2"},
    )
    assert (child / "prompt.md").read_text() == "p"
    assert read_manifest(child).parent == bundle_hash(base)


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def test_criteria_judgments():
    base, better, cheaper = (
        {"reward": 0.4, "cost_tokens": 100.0},
        {"reward": 0.6, "cost_tokens": 100.0},
        {"reward": 0.4, "cost_tokens": 80.0},
    )
    assert improve("reward").judge(base, better).accepted
    assert not improve("reward").judge(base, base).accepted  # needs strict climb
    assert minimize("cost_tokens", min_gain=0.1).judge(base, cheaper).accepted
    efficiency = guarded(minimize("cost_tokens", min_gain=0.1), [not_worse("reward")])
    verdict = efficiency.judge(base, cheaper)
    assert verdict.accepted and "guard reward" in verdict.reason
    worse = {"reward": 0.2, "cost_tokens": 50.0}
    assert not efficiency.judge(base, worse).accepted  # guard fires


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
    assert result.candidate == {"reward": 0.7}

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
    def rollout(bundle_dir, run_id):
        # Score lives in a custom file; the reward fn below reads it.
        strength = "strong" in (bundle_dir / "prompt.md").read_text()
        instance = tmp_path / "runs" / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        (instance / "out" / "result.json").write_text(
            json.dumps({"resolved": strength})
        )
        return [instance]

    def my_reward(run_dir):  # verl-style: one run dir in, one float out
        return (
            1.0
            if json.loads((run_dir / "out" / "result.json").read_text())["resolved"]
            else 0.0
        )

    def my_strategy(ctx):
        assert ctx.failures  # the observe rollout of the weak baseline
        return ctx.propose(
            kind="prompt",
            edits={"prompt.md": ctx.current("prompt.md") + " strong"},
            note="strengthen prompt",
            evidence=[ctx.failures[0].path],
        )

    lab = Lab(tmp_path, rollout=rollout, reward=my_reward, seed={"prompt.md": "weak"})
    report = lab.step(my_strategy)
    assert report.accepted and report.promoted_to
    assert "strong" in (resolve(tmp_path, "task") / "prompt.md").read_text()
    assert "ACCEPTED" in report.text and "accepted" in lab.history()

    # A strategy may also decline to propose.
    assert not lab.step(lambda ctx: None).proposed


def test_lab_rollback_returns_to_parent(tmp_path):
    def rollout(bundle_dir, run_id):
        instance = tmp_path / "runs" / run_id / "i1"
        (instance / "out").mkdir(parents=True)
        reward = 1.0 if "v2" in (bundle_dir / "prompt.md").read_text() else 0.0
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [instance]

    lab = Lab(tmp_path, rollout=rollout, seed={"prompt.md": "v1"})
    lab.step(
        lambda ctx: ctx.propose(kind="prompt", edits={"prompt.md": "v2"}, note="v2")
    )
    assert "v2" in (resolve(tmp_path, "task") / "prompt.md").read_text()
    assert "rolled back" in lab.rollback()
    assert "v1" in (resolve(tmp_path, "task") / "prompt.md").read_text()
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
