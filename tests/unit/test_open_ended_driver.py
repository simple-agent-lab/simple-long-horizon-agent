import random
import tempfile
import unittest
from pathlib import Path
import threading
from types import SimpleNamespace

from recipes.dgm.algorithm import archive, open_ended
from simple_agent_lab.evolution.components.criterion import valid_when
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Manifest, Proposal, Run, Slice, Verdict


def _seed(ws: Path) -> None:
    v = store.stage(
        ws,
        base=None,
        edits={"agent/agent_program.py": "x=1\n"},
        manifest=Manifest(producer="seed"),
    )
    store.promote(ws, v)


def _fake_components(ws, runs_by_hash):
    lock = threading.Lock()
    counter = {"n": 0}

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

        with lock:
            counter["n"] += 1
            n = counter["n"]
        return Proposal(
            edits={"agent/agent_program.py": f"x={n + 2}\n"}, note="m", kind="code"
        )

    return SimpleNamespace(
        rollout=rollout, reward=reward, strategy=strategy, criterion=valid_when()
    )


class OpenEndedDriverTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()

    def test_driver_admits_worse_but_valid_children(self):
        _seed(self.ws)
        comps = _fake_components(self.ws, runs_by_hash={})
        decisions = open_ended.run_evolution(
            self.ws, comps, Slice("s", ({"instance_id": "i1"},)), rounds=2, branches=2
        )
        self.assertEqual(len(decisions), 4)
        self.assertTrue(all(d.accepted for d in decisions))
        self.assertTrue(all(d.candidate.get("valid_parent") for d in decisions))

    def test_archive_can_select_a_noncurrent_valid_parent(self):
        _seed(self.ws)
        comps = _fake_components(self.ws, runs_by_hash={})
        open_ended.run_evolution(
            self.ws, comps, Slice("s", ({"instance_id": "i1"},)), rounds=1, branches=2
        )
        nodes = archive.nodes(self.ws)
        valid = [n for n in nodes if n.valid_parent and "reward" in n.scores]
        self.assertGreaterEqual(len(valid), 1)
        chosen = archive.select_parent(valid, method="random", rng=random.Random(0))
        self.assertIsInstance(chosen, str)
        self.assertTrue(chosen)

    def test_branch_baseline_is_the_proposal_base(self):
        _seed(self.ws)
        root = store.current(self.ws)
        parent = store.stage(
            self.ws,
            base=root,
            edits={"agent/agent_program.py": "x=9\n"},
            manifest=Manifest(parent=root.hash, producer="fixture"),
        )

        def rollout(version, _slice):
            reward = {root.hash: 0.1, parent.hash: 0.6}.get(version.hash, 0.8)
            run_dir = self.ws / "runs" / version.hash / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(
                '{"reward": %s}' % reward, encoding="utf-8"
            )
            return [Run(run_dir)]

        def strategy(_ctx):
            return Proposal(
                base=parent.hash,
                edits={"agent/agent_program.py": "x=10\n"},
                note="branch",
            )

        def reward(run):
            return run.reward if run.reward is not None else 0.0

        def criterion(baseline, candidate):
            base_mean = next(iter(baseline.values()))["reward"]
            cand_mean = next(iter(candidate.values()))["reward"]
            return Verdict(
                True,
                "accepted",
                {"reward": cand_mean - base_mean, "valid_parent": 1.0},
            )

        decisions = open_ended.run_round(
            self.ws,
            SimpleNamespace(
                rollout=rollout,
                reward=reward,
                strategy=strategy,
                criterion=criterion,
            ),
            Slice("s", ({"instance_id": "i1"},)),
            branches=1,
        )

        self.assertEqual(decisions[0].baseline["hash"], parent.hash)
        self.assertEqual(decisions[0].baseline["scores"]["reward"], 0.6)

    def test_duplicate_candidates_share_one_rollout_and_decision(self):
        _seed(self.ws)
        rollout_counts: dict[str, int] = {}
        lock = threading.Lock()

        def rollout(version, _slice):
            with lock:
                rollout_counts[version.hash] = rollout_counts.get(version.hash, 0) + 1
            run_dir = self.ws / "runs" / version.hash / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(
                '{"reward": 0.5}', encoding="utf-8"
            )
            return [Run(run_dir)]

        def strategy(_ctx):
            return Proposal(
                edits={"agent/agent_program.py": "x=2\n"},
                note="same candidate",
                kind="code",
            )

        components = SimpleNamespace(
            rollout=rollout,
            reward=lambda run: run.reward if run.reward is not None else 0.0,
            strategy=strategy,
            criterion=valid_when(),
        )

        decisions = open_ended.run_round(
            self.ws,
            components,
            Slice("s", ({"instance_id": "i1"},)),
            branches=3,
        )

        self.assertEqual(len(decisions), 1)
        candidate_hash = decisions[0].candidate["hash"]
        self.assertEqual(rollout_counts[candidate_hash], 1)


if __name__ == "__main__":
    unittest.main()
