import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evals.suites.swebench import evolving_rollout as er
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Manifest, Run, Version


class EvolvingRolloutTest(unittest.TestCase):
    def test_reward_from_result_prefers_resolved_then_score_then_reward(self):
        self.assertEqual(er.reward_from_result({"resolved": True}), 1.0)
        self.assertEqual(er.reward_from_result({"resolved": False}), 0.0)
        self.assertEqual(er.reward_from_result({"score": 0.5}), 0.5)
        self.assertEqual(er.reward_from_result({"reward": 0.5}), 0.5)
        self.assertEqual(er.reward_from_result({}), 0.0)

    def test_performance_layout_paths(self):
        layout = er.PerformanceLayout(Path("/tmp/out"), "run1")
        self.assertEqual(layout.run_root, Path("/tmp/out/run1"))
        self.assertEqual(layout.evolution_workspace, Path("/tmp/out/run1/evolution"))
        self.assertEqual(layout.swebench_runs, Path("/tmp/out/run1/swebench_runs"))
        self.assertEqual(layout.official, Path("/tmp/out/run1/official"))
        self.assertEqual(
            layout.generation_metrics,
            Path("/tmp/out/run1/generation_metrics.jsonl"),
        )
        self.assertTrue(str(layout.predictions).endswith("run1_predictions.jsonl"))

    def test_load_dataset_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.jsonl"
            p.write_text(
                json.dumps({"instance_id": "i1"})
                + "\n"
                + json.dumps({"instance_id": "i2"})
                + "\n",
                encoding="utf-8",
            )
            rows = er.load_dataset(p)
        self.assertEqual([r["instance_id"] for r in rows], ["i1", "i2"])

    def test_collect_and_official_commands(self):
        layout = er.PerformanceLayout(Path("evals/out/h"), "run-1")
        collect = er.collect_predictions_command(
            layout, dataset_name=er.DEFAULT_DATASET, model_name="m-best"
        )
        official = er.official_eval_command(
            layout,
            dataset_name=er.DEFAULT_DATASET,
            instance_ids=("sympy__sympy-23824",),
            max_workers=2,
        )
        self.assertIn("evals/swebench/evaluate_predictions.py", collect)
        self.assertIn("--collect-predictions", collect)
        self.assertIn(str(layout.swebench_runs), collect)
        self.assertIn("--run-official", official)
        self.assertIn("--instance-ids", official)
        self.assertIn("sympy__sympy-23824", official)

    def test_generation_metric_record_rates(self):
        rec = er.generation_metric_record(
            generation=2,
            version_hash="abc",
            parent_hash="root",
            parent_selection="best",
            decision_outcome="accepted",
            runs=[
                {"instance_id": "i1", "reward": 1.0, "patch_chars": 10, "tokens": 7},
                {"instance_id": "i2", "reward": 0.0, "patch_chars": 0, "tokens": 3},
            ],
        )
        self.assertEqual(rec["total"], 2)
        self.assertEqual(rec["resolved"], 1)
        self.assertEqual(rec["patch_valid"], 1)
        self.assertEqual(rec["tokens"], 10)
        self.assertEqual(rec["resolved_rate"], 0.5)

    def test_seed_files_includes_agent_package(self):
        seed = er.seed_files(
            model="gpt-test",
            api_kind="openai-chat",
            base_url="https://example.test/v1",
        )
        self.assertIn("agent/agent_program.py", seed)
        self.assertIn("build_agent", seed["agent/agent_program.py"])
        self.assertIn("provider.json", seed)
        self.assertIn("README.md", seed)
        provider = json.loads(seed["provider.json"])
        self.assertEqual(provider["model"], "gpt-test")
        self.assertEqual(provider["api"], "openai-chat")
        self.assertEqual(provider["base_url"], "https://example.test/v1")
        self.assertEqual(provider["api_key_env"], er.OPENAI_AUTH_ENV)

    def test_version_package_artifacts_contains_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = store.stage(
                Path(tmp),
                base=None,
                edits={"agent/agent_program.py": "def build_agent(): ...\n"},
                manifest=Manifest(producer="seed"),
            )
            art = er.version_package_artifacts(v)
        files = json.loads(art[AGENT_PACKAGE_KEY].decode("utf-8"))
        self.assertIn("agent_program.py", files)

    def test_package_files_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "version"
            version_dir.mkdir()
            files = er.package_files(Version(version_dir))
        self.assertIn("agent_program.py", files)
        self.assertIn("build_agent", files["agent_program.py"])

    def test_apply_eval_score_updates_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run" / "case-1"
            out = run_dir / "out"
            out.mkdir(parents=True)
            (out / "result.json").write_text('{"model_patch": "diff"}\n')
            er.apply_eval_score(
                Run(run_dir),
                {
                    "passed": True,
                    "score": 1.0,
                    "reason": "resolved",
                    "metrics": {"resolved": True, "status": "resolved"},
                },
            )
            result = json.loads((out / "result.json").read_text())
        self.assertTrue(result["resolved"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["score"], 1.0)


class EnsureRolloutArtifactsTest(unittest.TestCase):
    def _make_run(self, base, instance_id, *, complete):
        run_dir = base / instance_id
        (run_dir / "out").mkdir(parents=True)
        if complete:
            (run_dir / "out" / "result.json").write_text(
                json.dumps({"status": "eval_script_ran"}), encoding="utf-8"
            )
        return Run(run_dir)

    def test_fails_when_result_json_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run" / "case-1"
            (run_dir / "out").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "missing result.json"):
                er.ensure_rollout_artifacts([Run(run_dir)])

    def test_tolerates_a_few_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "run"
            runs = [
                self._make_run(base, "case-1", complete=True),
                self._make_run(base, "case-2", complete=True),
                self._make_run(base, "case-3", complete=True),
                self._make_run(base, "case-4", complete=False),
            ]
            er.ensure_rollout_artifacts(runs)

    def test_raises_on_systemic_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "run"
            runs = [
                self._make_run(base, "case-1", complete=True),
                self._make_run(base, "case-2", complete=False),
                self._make_run(base, "case-3", complete=False),
                self._make_run(base, "case-4", complete=False),
            ]
            with self.assertRaisesRegex(RuntimeError, "1/4|completion floor"):
                er.ensure_rollout_artifacts(runs)

    def test_grade_reuse_runs_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "run"
            incomplete = self._make_run(base, "case-1", complete=False)
            er.grade_reuse_runs(
                [incomplete],
                ({"instance_id": "case-1"},),
                dataset_name="ds",
                model_name="m",
            )
            self.assertFalse((incomplete.dir / "out" / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
