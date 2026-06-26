import random
from io import StringIO
import tempfile
import unittest
from pathlib import Path
import threading
from types import SimpleNamespace

from recipes.dgm import evolve as dgm_evolve
from recipes.dgm.algorithm import archive, open_ended
from simple_agent_lab.evolution.components.criterion import valid_when
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.progress import ProgressReporter
from simple_agent_lab.evolution.types import (
    Manifest,
    Proposal,
    Run,
    Slice,
    Verdict,
    Version,
)


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

    def test_driver_records_candidate_metadata_and_uses_valid_parent(self):
        _seed(self.ws)

        def rollout(version, _slice):
            run_dir = self.ws / "runs" / version.hash / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(
                '{"reward": 0.0}', encoding="utf-8"
            )
            return [Run(run_dir)]

        def strategy(_ctx):
            return Proposal(
                edits={"agent/agent_program.py": "x=22\n"},
                note="metadata",
                kind="code",
            )

        components = SimpleNamespace(
            rollout=rollout,
            reward=lambda _run: {"reward": 0.0, "valid_parent": 1.0},
            strategy=strategy,
            criterion=valid_when(),
            candidate_metadata=lambda _runs: {
                "valid_parent": False,
                "agent_build_failed": 1,
                "completed": 0,
            },
        )

        decisions = open_ended.run_round(
            self.ws,
            components,
            Slice("s", ({"instance_id": "i1"},)),
            branches=1,
        )

        self.assertFalse(decisions[0].accepted)
        self.assertFalse(decisions[0].candidate["valid_parent"])
        self.assertEqual(decisions[0].candidate["diagnostics"]["agent_build_failed"], 1)

    def test_progress_reports_round_candidate_decision_and_promotion(self):
        _seed(self.ws)
        stream = StringIO()
        progress = ProgressReporter(stream=stream)

        decisions = open_ended.run_evolution(
            self.ws,
            _fake_components(self.ws, runs_by_hash={}),
            Slice("s", ({"instance_id": "i1"},)),
            rounds=1,
            branches=1,
            progress=progress,
        )

        self.assertEqual(len(decisions), 1)
        output = stream.getvalue()
        self.assertIn("[progress] dgm round start index=1 total=1 branches=1", output)
        self.assertIn("[progress] dgm candidate staged branch=1", output)
        self.assertIn("[progress] decision accepted", output)
        self.assertIn("valid_parent=true", output)
        self.assertIn("[progress] dgm promote", output)
        self.assertIn("[progress] dgm round complete index=1 decisions=1", output)


class DgmRecipeProgressTest(unittest.TestCase):
    def test_run_start_progress_includes_dgm_configuration(self):
        stream = StringIO()
        progress = ProgressReporter(stream=stream)
        args = SimpleNamespace(
            run_id="dgm-real",
            output_root="/tmp/dgm-out",
            config="configs/dgm_swebench.yaml",
            parent_selection="best",
            api_kind="openai-chat",
            model_name="gpt-test",
        )

        dgm_evolve._print_progress_run_start(
            progress,
            args,
            rounds=4,
            branches=3,
            meta_workers=2,
            global_workers=6,
            train_count=60,
            heldout_count=12,
        )

        output = stream.getvalue()
        self.assertIn("[progress] run start id=dgm-real", output)
        self.assertIn("rounds=4", output)
        self.assertIn("branches=3", output)
        self.assertIn("meta_workers=2", output)
        self.assertIn("parallel=6", output)
        self.assertIn("train=60", output)
        self.assertIn("heldout=12", output)
        self.assertIn("parent_selection=best", output)
        self.assertIn("model=openai-chat", output)

    def test_skipped_baseline_heldout_progress_names_reason_and_count(self):
        stream = StringIO()
        progress = ProgressReporter(stream=stream)
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "abc123"
            version_dir.mkdir()
            record = dgm_evolve.skipped_heldout_record(
                Version(version_dir),
                [{"instance_id": "i1"}, {"instance_id": "i2"}],
                label="baseline",
            )

        dgm_evolve._print_progress_heldout_skipped(
            progress, record, reason="skip_baseline_heldout"
        )

        self.assertIn(
            "[progress] heldout skipped label=baseline "
            "reason=skip_baseline_heldout version=abc123 count=2",
            stream.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
