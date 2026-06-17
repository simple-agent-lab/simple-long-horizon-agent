import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimpleRecipeSmokeTest(unittest.TestCase):
    def test_simple_parser_builds_and_is_dry_by_default(self):
        mod = _load(ROOT / "recipes" / "simple" / "evolve.py")
        args = mod.build_parser().parse_args(
            ["--train-dataset", "t.jsonl", "--test-dataset", "e.jsonl", "--run-id", "x"]
        )
        self.assertFalse(args.execute)
        self.assertEqual(args.rounds, 4)
        self.assertEqual(args.promotion_tolerance, 0.0)
        self.assertEqual(args.uv_binary, "")

    def test_simple_heldout_run_id_uses_version_and_slice(self):
        mod = _load(ROOT / "recipes" / "simple" / "evolve.py")
        from simple_agent_lab.evolution.types import Version

        with tempfile.TemporaryDirectory() as tmp:
            vd = Path(tmp) / "simplev"
            (vd / "agent").mkdir(parents=True)
            (vd / "agent" / "agent_program.py").write_text(
                "def build_agent(): ...\n", encoding="utf-8"
            )
            rid = mod.heldout_run_id(Version(vd), ({"instance_id": "test-1"},))
        self.assertTrue(rid.startswith("simplev-"))


class DgmRecipeSmokeTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "evolve.py")

    def test_parser_exposes_dgm_knobs(self):
        ns = self.mod.build_parser().parse_args(
            [
                "--run-id",
                "x",
                "--train-dataset",
                "t.jsonl",
                "--test-dataset",
                "e.jsonl",
                "--parent-selection",
                "best",
                "--branches",
                "4",
            ]
        )
        self.assertEqual(ns.parent_selection, "best")
        self.assertEqual(ns.branches, 4)
        self.assertFalse(ns.execute)

    def test_pick_best_node_selects_highest_valid(self):
        from recipes.dgm import archive

        nodes = (
            archive.ArchiveNode(hash="a", scores={"reward": 0.5}),
            archive.ArchiveNode(hash="b", scores={"reward": 0.9}),
            archive.ArchiveNode(hash="d", scores={"reward": 1.0}, valid_parent=False),
            archive.ArchiveNode(hash="e", scores={}),
        )
        best = self.mod.pick_best_node(nodes)
        self.assertIsNotNone(best)
        self.assertEqual(best.hash, "b")

    def test_pick_best_node_none_when_empty(self):
        self.assertIsNone(self.mod.pick_best_node(()))

    def test_score_reads_reward_from_candidate(self):
        self.assertEqual(self.mod._score({"scores": {"reward": 0.5}}), 0.5)
        self.assertEqual(self.mod._score({}), 0.0)
        self.assertEqual(self.mod._score({"scores": "bad"}), 0.0)

    def test_dgm_admission_rejects_agent_package_fallback_reward(self):
        criterion = self.mod.dgm_admission_criterion("reward")
        verdict = criterion({"i1": {"reward": 1.0}}, {"i1": {"reward": -1.0}})
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.deltas["valid_parent"], 0.0)

    def test_dgm_admission_accepts_worse_but_valid_zero_reward(self):
        criterion = self.mod.dgm_admission_criterion("reward")
        verdict = criterion({"i1": {"reward": 1.0}}, {"i1": {"reward": 0.0}})
        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.deltas["valid_parent"], 1.0)

    def test_resolve_schedule_rounds_wins_over_generations(self):
        ns = self.mod.build_parser().parse_args(
            [
                "--run-id",
                "x",
                "--train-dataset",
                "t",
                "--test-dataset",
                "e",
                "--rounds",
                "6",
                "--generations",
                "9",
                "--branches",
                "2",
            ]
        )
        rounds, branches, meta = self.mod.resolve_schedule(ns)
        self.assertEqual((rounds, branches, meta), (6, 2, 2))

    def test_resolve_schedule_falls_back_to_generations_then_default(self):
        base = ["--run-id", "x", "--train-dataset", "t", "--test-dataset", "e"]
        ns_gen = self.mod.build_parser().parse_args(base + ["--generations", "7"])
        self.assertEqual(self.mod.resolve_schedule(ns_gen)[0], 7)
        ns_def = self.mod.build_parser().parse_args(base)
        rounds, branches, meta = self.mod.resolve_schedule(ns_def)
        self.assertEqual(rounds, 4)
        self.assertEqual(meta, branches)

    def test_heldout_run_id_uses_version_and_slice(self):
        from simple_agent_lab.evolution.types import Version

        with tempfile.TemporaryDirectory() as tmp:
            vd = Path(tmp) / "abc123"
            (vd / "agent").mkdir(parents=True)
            (vd / "agent" / "agent_program.py").write_text(
                "def build_agent(): ...\n", encoding="utf-8"
            )
            rid = self.mod.heldout_run_id(Version(vd), ({"instance_id": "test-1"},))
        self.assertTrue(rid.startswith("abc123-"))


class RecordHeldoutGenerationTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "evolve.py")
        from evals.swebench import evolution_adapter as er

        self.er = er

    def _eval_rows(self):
        return [
            {
                "type": "eval_result",
                "scorer": "swebench",
                "passed": True,
                "score": 1.0,
                "reason": "resolved",
                "metrics": {"instance_id": "pass-0", "patch_chars": 50},
            },
            {
                "type": "eval_result",
                "scorer": "swebench",
                "passed": False,
                "score": 0.0,
                "reason": "unresolved",
                "metrics": {"instance_id": "fail-0", "patch_chars": 0},
            },
        ]

    def test_writes_one_row_from_heldout_eval_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "run")
            layout.create()
            layout.eval_results.write_text(
                "\n".join(json.dumps(r) for r in self._eval_rows()),
                encoding="utf-8",
            )
            record = self.mod.record_heldout_generation(
                layout,
                SimpleNamespace(hash="v1", parent="v0"),
                parent_selection="best",
            )
            self.assertEqual(record["resolved"], 1)
            self.assertEqual(record["total"], 2)
            self.assertEqual(record["resolved_rate"], 0.5)
            self.assertEqual(record["test_resolved_rate"], 0.5)
            self.assertEqual(record["version"], "v1")
            self.assertEqual(record["parent"], "v0")
            self.assertEqual(record["parent_selection"], "best")

            from simple_agent_lab.trace.jsonl import read_jsonl

            self.assertTrue(layout.generation_metrics.is_file())
            written = read_jsonl(layout.generation_metrics)
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["resolved_rate"], 0.5)
            self.assertEqual(written[0]["test_resolved_rate"], 0.5)

    def test_skips_write_when_eval_results_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "run")
            layout.create()
            record = self.mod.record_heldout_generation(
                layout,
                SimpleNamespace(hash="v1", parent="v0"),
                parent_selection="best",
            )
            self.assertEqual(record, {})
            self.assertFalse(layout.generation_metrics.is_file())

    def test_report_summarizes_written_generation_metrics(self):
        report = _load(ROOT / "recipes" / "dgm" / "report.py")
        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "run")
            layout.create()
            layout.eval_results.write_text(
                "\n".join(json.dumps(r) for r in self._eval_rows()),
                encoding="utf-8",
            )
            self.mod.record_heldout_generation(
                layout,
                SimpleNamespace(hash="v1", parent="v0"),
                parent_selection="best",
            )
            (layout.run_root / "decisions.jsonl").write_text(
                json.dumps(
                    {
                        "id": "d-000001",
                        "accepted": True,
                        "candidate": {"hash": "v1", "scores": {"reward": 0.5}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = report.summarize(layout.run_root)
        self.assertEqual(summary["generations"], 1)
        self.assertEqual(summary["best_resolved_rate"], 0.5)
        self.assertEqual(summary["monitor"]["latest_test_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
