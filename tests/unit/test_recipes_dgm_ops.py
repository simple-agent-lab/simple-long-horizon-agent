import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(n_pass, n_fail):
    rows = []
    for i in range(n_pass):
        rows.append({"instance_id": f"pass-{i}", "resolved": True})
    for i in range(n_fail):
        rows.append({"instance_id": f"fail-{i}", "resolved": False})
    return rows


class SelectHeadroomTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "baseline.py")

    def test_balances_passes_and_fails(self):
        chosen = self.mod.select_headroom(
            _baseline(20, 20), want=10, pass_fraction=0.5, seed=1
        )
        self.assertEqual(len(chosen), 10)
        self.assertEqual(sum(1 for c in chosen if c.startswith("pass-")), 5)

    def test_backfills_when_one_class_short(self):
        chosen = self.mod.select_headroom(
            _baseline(3, 30), want=20, pass_fraction=0.45, seed=1
        )
        self.assertEqual(len(chosen), 20)
        self.assertEqual(len(set(chosen)), 20)

    def test_caps_at_available(self):
        chosen = self.mod.select_headroom(_baseline(2, 2), want=10, seed=0)
        self.assertEqual(len(chosen), 4)


class SplitChosenTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "baseline.py")

    def test_disjoint_full_records(self):
        pool = [
            {"instance_id": f"i{i}", "problem_statement": f"p{i}"} for i in range(10)
        ]
        chosen = [f"i{i}" for i in range(10)]
        train, test = self.mod.split_chosen(
            chosen, pool, train_size=4, test_size=3, seed=2
        )
        train_ids = {r["instance_id"] for r in train}
        test_ids = {r["instance_id"] for r in test}
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 3)
        self.assertEqual(train_ids & test_ids, set())
        self.assertIn("problem_statement", train[0])

    def test_raises_when_not_enough(self):
        with self.assertRaisesRegex(ValueError, "need 5 headroom"):
            self.mod.split_chosen(
                ["i0"], [{"instance_id": "i0"}], train_size=3, test_size=2, seed=0
            )


class BaselineMainSafetyTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "baseline.py")

    def test_rejects_unsafe_run_id_before_pool_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unsafe run id"):
                self.mod.main(
                    [
                        "--run-id",
                        "../victim",
                        "--output-root",
                        str(root / "out"),
                        "--pool",
                        str(root / "missing-pool.jsonl"),
                        "--train-out",
                        str(root / "train.jsonl"),
                        "--test-out",
                        str(root / "test.jsonl"),
                    ]
                )

    def test_measure_pool_rejects_unsafe_run_id_before_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unsafe run id"):
                self.mod.measure_pool(
                    [],
                    run_id="../victim",
                    output_root=root / "out",
                    dataset_name="dataset",
                    concurrency=1,
                    api_kind="openai-chat",
                    max_turns=1,
                    model_name="model",
                )


class ReportSummarizeTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "report.py")

    def test_summarizes_best_generation_and_selectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            (run_root / "generation_metrics.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "version": "a",
                        "parent_selection": "latest",
                        "resolved_rate": 0.25,
                        "resolved": 1,
                        "total": 4,
                        "tokens": 100,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "generation": 2,
                        "version": "b",
                        "parent_selection": "score_child_prop",
                        "resolved_rate": 0.5,
                        "resolved": 2,
                        "total": 4,
                        "tokens": 200,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = self.mod.summarize(run_root)
        self.assertEqual(summary["best_generation"], 2)
        self.assertEqual(summary["best_version"], "b")
        self.assertEqual(summary["best_resolved_rate"], 0.5)
        self.assertEqual(summary["selector_counts"]["score_child_prop"], 1)

    def test_monitor_summarizes_decisions_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            (run_root / "generation_metrics.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "version": "v1",
                        "parent_selection": "best",
                        "resolved_rate": 0.5,
                        "test_resolved_rate": 0.25,
                        "tokens": 10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / "decisions.jsonl").write_text(
                json.dumps(
                    {
                        "id": "d-000001",
                        "accepted": True,
                        "candidate": {"hash": "v1", "scores": {"reward": 0.5}},
                        "runs": {"candidate": "run-containing-test-1"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "d-000002",
                        "accepted": False,
                        "candidate": {"hash": "v2", "scores": {"reward": 0.25}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            test_dataset = run_root / "test.jsonl"
            test_dataset.write_text(json.dumps({"instance_id": "test-1"}) + "\n")
            summary = self.mod.summarize(run_root, test_dataset=test_dataset)
        self.assertEqual(summary["monitor"]["decision_count"], 2)
        self.assertEqual(summary["monitor"]["accepted"], 1)
        self.assertEqual(summary["monitor"]["current_version"], "v1")
        self.assertTrue(summary["monitor"]["test_touched_before_final_scoring"])


if __name__ == "__main__":
    unittest.main()
