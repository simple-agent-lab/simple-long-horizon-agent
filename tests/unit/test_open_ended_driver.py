import random
from pathlib import Path
from types import SimpleNamespace

from simple_agent_lab.evolution import archive
from simple_agent_lab.evolution.components.criterion import valid_when
from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.types import Manifest, Run, Slice
from scripts.evolution_recipes.hyperagents import open_ended


def _seed(ws: Path) -> None:
    v = store.stage(ws, base=None, edits={"agent/agent_program.py": "x=1\n"},
                    manifest=Manifest(producer="seed"))
    store.promote(ws, v)


def _fake_components(ws, runs_by_hash):
    def rollout(version, slice_):
        run_dir = ws / "runs" / version.hash / "i1"
        (run_dir / "out").mkdir(parents=True, exist_ok=True)
        (run_dir / "out" / "result.json").write_text(
            '{"reward": %s}' % runs_by_hash.get(version.hash, 0.0)
        )
        return [Run(run_dir)]

    def reward(run):
        return run.reward if run.reward is not None else 0.0

    def strategy(ctx):
        from simple_agent_lab.evolution.types import Proposal
        n = len(tuple(log.read(ctx.workspace)))
        return Proposal(edits={"agent/agent_program.py": f"x={n+2}\n"},
                        note="m", kind="code")

    return SimpleNamespace(
        rollout=rollout, reward=reward, strategy=strategy, criterion=valid_when()
    )


def test_driver_admits_worse_but_valid_children(tmp_path: Path):
    _seed(tmp_path)
    comps = _fake_components(tmp_path, runs_by_hash={})  # all reward 0.0
    decisions = open_ended.run_evolution(
        tmp_path, comps, Slice("s", ({"instance_id": "i1"},)),
        rounds=2, branches=2,
    )
    assert len(decisions) == 4
    assert all(d.accepted for d in decisions)
    assert all(d.candidate.get("valid_parent") for d in decisions)


def test_archive_can_select_a_noncurrent_valid_parent(tmp_path: Path):
    _seed(tmp_path)
    comps = _fake_components(tmp_path, runs_by_hash={})
    open_ended.run_evolution(
        tmp_path, comps, Slice("s", ({"instance_id": "i1"},)),
        rounds=1, branches=2,
    )
    nodes = archive.nodes(tmp_path)
    valid = [n for n in nodes if n.valid_parent and "reward" in n.scores]
    assert len(valid) >= 1
    chosen = archive.select_parent(valid, method="random", rng=random.Random(0))
    assert isinstance(chosen, str) and chosen
