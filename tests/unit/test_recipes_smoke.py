import importlib.util
import json
import os
import subprocess
import sys
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


class OfficialHeldoutArtifactsTest(unittest.TestCase):
    def setUp(self):
        from recipes.dgm import swebench as er

        self.er = er

    def test_official_artifacts_are_scoped_by_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "demo")
            baseline = self.er.official_artifacts(layout, "baseline")
            final = self.er.official_artifacts(layout, "final")

        self.assertNotEqual(baseline.predictions, final.predictions)
        self.assertEqual(baseline.predictions.name, "baseline_predictions.jsonl")
        self.assertEqual(final.eval_results.name, "eval_results.jsonl")
        self.assertIn("baseline", baseline.harness.as_posix())
        self.assertIn("final", final.harness.as_posix())

    def test_official_commands_accept_scoped_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "demo")
            artifacts = self.er.official_artifacts(layout, "baseline")
            collect = self.er.collect_predictions_command(
                layout,
                dataset_name="dataset",
                model_name="model",
                source_run_id="version-slice",
                predictions=artifacts.predictions,
            )
            official = self.er.official_eval_command(
                layout,
                dataset_name="dataset",
                instance_ids=("i1",),
                max_workers=2,
                predictions=artifacts.predictions,
                eval_results=artifacts.eval_results,
                official_output_dir=artifacts.harness,
                run_id=artifacts.run_id,
            )

        self.assertIn(str(artifacts.predictions), collect)
        self.assertIn(str(artifacts.predictions), official)
        self.assertIn(str(artifacts.eval_results), official)
        self.assertIn(str(artifacts.harness), official)
        self.assertIn(artifacts.run_id, official)

    def test_summarizes_official_eval_results(self):
        from simple_agent_lab.trace.jsonl import write_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_results.jsonl"
            write_jsonl(
                path,
                [
                    {"passed": True, "score": 1.0},
                    {"passed": False, "score": 0.0},
                    {"passed": True, "score": 1.0},
                ],
            )

            summary = self.er.summarize_official_eval_results(path)

        self.assertEqual(summary["resolved"], 2)
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["resolved_rate"], 2 / 3)


class RecipeLayoutTest(unittest.TestCase):
    def test_recipes_have_clear_runtime_algorithm_and_ops_slots(self):
        self.assertTrue((ROOT / "recipes" / "runtime.py").is_file())
        self.assertFalse((ROOT / "recipes" / "_shared.py").exists())
        self.assertTrue(
            (ROOT / "recipes" / "dgm" / "algorithm" / "archive.py").is_file()
        )
        self.assertTrue(
            (ROOT / "recipes" / "dgm" / "algorithm" / "open_ended.py").is_file()
        )
        self.assertTrue(
            (ROOT / "recipes" / "dgm" / "algorithm" / "repo_edits.py").is_file()
        )
        self.assertTrue((ROOT / "recipes" / "dgm" / "ops" / "baseline.py").is_file())
        self.assertTrue((ROOT / "recipes" / "dgm" / "ops" / "report.py").is_file())
        self.assertTrue((ROOT / "recipes" / "dgm" / "swebench.py").is_file())
        self.assertFalse((ROOT / "recipes" / "dgm" / "archive.py").exists())
        self.assertFalse((ROOT / "recipes" / "dgm" / "baseline.py").exists())
        self.assertFalse((ROOT / "recipes" / "__init__.py").exists())
        self.assertFalse((ROOT / "recipes" / "simple" / "__init__.py").exists())
        self.assertFalse((ROOT / "recipes" / "dgm" / "__init__.py").exists())
        self.assertFalse((ROOT / "evals" / "swebench" / "self_evolving.py").exists())
        self.assertFalse(
            (ROOT / "evals" / "swebench" / "evolution_adapter.py").exists()
        )


class DefaultConfigDatasetPathTest(unittest.TestCase):
    def _assert_real_jsonl(self, path: str):
        data_path = Path(path)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        self.assertTrue(data_path.is_file(), path)
        self.assertNotIn("configs/examples", data_path.as_posix())

    def test_simple_default_config_uses_real_dataset_paths(self):
        from simple_agent_lab.evolution.config import load_self_evolving_config

        config = load_self_evolving_config(ROOT / "configs" / "simple_swebench.yaml")

        self._assert_real_jsonl(config.instances.train.path)
        self.assertIsNotNone(config.instances.heldout)
        self._assert_real_jsonl(config.instances.heldout.path)

    def test_dgm_default_config_uses_real_dataset_paths(self):
        from recipes.dgm.config import load_dgm_config

        config = load_dgm_config(ROOT / "configs" / "dgm_swebench.yaml")

        self._assert_real_jsonl(config.dataset.train_path)
        self._assert_real_jsonl(config.dataset.test_path)

    def test_ahe_default_config_uses_real_dataset_paths(self):
        from simple_agent_lab.evolution.config import load_self_evolving_config

        config = load_self_evolving_config(ROOT / "configs" / "ahe_swebench.yaml")

        self._assert_real_jsonl(config.instances.train.path)
        self.assertIsNotNone(config.instances.heldout)
        self._assert_real_jsonl(config.instances.heldout.path)


class SimpleRecipeSmokeTest(unittest.TestCase):
    def test_simple_recipe_runs_generic_dry_run_with_registered_swebench(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            train.write_text('{"instance_id": "sympy__sympy-1"}\n', encoding="utf-8")
            config = root / "simple.yaml"
            config.write_text(
                f"""
run:
  id: temp-simple
  output_root: {root / "out"}
  execute: false
  reset: false
  dotenv: .env
suite:
  name: swebench
surface:
  name: python_agent_package
  editable_components: [everything]
  artifact_key: input/agent_package.json
  default: simple_agent_package
instances:
  train:
    id: train
    path: {train}
execution:
  backend:
    name: local_docker
  store:
    name: local_dir
  parallel: 1
  max_turns: 3
model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
strategy:
  name: model_program
  args:
    system_prompt: demo
evolution:
  algorithm: simple
  rounds: 2
  criterion:
    name: promote_not_worse
    args:
      dim: reward
evaluation:
  baseline_heldout: false
  final_heldout: false
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: false
""".lstrip(),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "recipes/simple/evolve.py",
                    "--config",
                    str(config),
                    "--run-id",
                    "override-simple",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("run id: override-simple", result.stdout)
        self.assertIn("suite: swebench", result.stdout)
        self.assertIn("surface: python_agent_package", result.stdout)
        self.assertIn("editable components: everything", result.stdout)

    def test_simple_recipe_default_config_runs_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "recipes/simple/evolve.py",
                "--run-id",
                "default-simple-smoke",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("run id: default-simple-smoke", result.stdout)


class AheRecipeSmokeTest(unittest.TestCase):
    def test_ahe_execute_writes_round_ledger_artifacts(self):
        mod = _load(ROOT / "recipes" / "ahe" / "evolve.py")

        class FakeExperiment:
            def __init__(self, workspace: Path, decisions):
                self.workspace = workspace
                self._decisions = list(decisions)

            def step(self, _strategy):
                return self._decisions.pop(0) if self._decisions else None

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "out" / "ahe-run"
            workspace = run_root / "evolution"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_reward_run(run_root / "runs" / "baseline-run", "i1", 0.0)
            self._write_reward_run(run_root / "runs" / "baseline-run", "i2", 1.0)
            self._write_reward_run(run_root / "runs" / "candidate-run", "i1", 1.0)
            self._write_reward_run(run_root / "runs" / "candidate-run", "i2", 0.0)
            round_path = run_root / "ahe" / "rounds" / "round_001"
            round_path.mkdir(parents=True, exist_ok=True)
            (round_path / "change_manifest.json").write_text(
                json.dumps(
                    {
                        "round": 1,
                        "changes": [
                            {
                                "id": "chg-1",
                                "predicted_fixes": ["i1"],
                                "risk_tasks": ["i2"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            decision = self._make_decision(
                accepted=True,
                reason="candidate fixed one task but regressed another",
                baseline_run_id="baseline-run",
                candidate_run_id="candidate-run",
            )
            config = SimpleNamespace(
                run=SimpleNamespace(id="ahe-run"),
                evolution=SimpleNamespace(rounds=1),
                evaluation=SimpleNamespace(
                    baseline_heldout=False,
                    final_heldout=False,
                    heldout_every_rounds=0,
                    repeats=1,
                    official_scoring=False,
                ),
            )
            built = SimpleNamespace(
                strategy=object(),
                reward=lambda run: run.reward if run.reward is not None else 0.0,
                heldout=None,
                experiment=FakeExperiment(workspace, [decision]),
            )

            decisions, report_path = mod._run_ahe_execute(config, built)

            self.assertEqual(len(decisions), 1)
            self.assertIsNone(report_path)
            self.assertTrue((run_root / "ahe" / "task_history.json").is_file())
            self.assertTrue((run_root / "ahe" / "best_ever.json").is_file())
            self.assertTrue((run_root / "ahe" / "history.md").is_file())
            self.assertTrue((round_path / "change_evaluation.json").is_file())

            task_history = json.loads(
                (run_root / "ahe" / "task_history.json").read_text(encoding="utf-8")
            )
            best_ever = json.loads(
                (run_root / "ahe" / "best_ever.json").read_text(encoding="utf-8")
            )
            change_eval = json.loads(
                (round_path / "change_evaluation.json").read_text(encoding="utf-8")
            )
            history = (run_root / "ahe" / "history.md").read_text(encoding="utf-8")

        self.assertEqual(task_history["i1"][0]["round"], 1)
        self.assertEqual(task_history["i1"][0]["scores"], {"reward": 1.0})
        self.assertTrue(task_history["i1"][0]["passed"])
        self.assertEqual(best_ever["version"], "candidate-hash")
        self.assertEqual(change_eval["fixed_tasks"], ["i1"])
        self.assertEqual(change_eval["regressed_tasks"], ["i2"])
        self.assertIn("round 1", history)
        self.assertIn("candidate-hash", history)
        self.assertIn("accepted", history)

    def test_ahe_best_ever_keeps_baseline_when_candidate_is_rejected(self):
        mod = _load(ROOT / "recipes" / "ahe" / "evolve.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            round_path = run_root / "ahe" / "rounds" / "round_001"
            round_path.mkdir(parents=True)
            (round_path / "change_manifest.json").write_text(
                json.dumps({"round": 1, "changes": []}),
                encoding="utf-8",
            )
            self._write_reward_run(run_root / "runs" / "baseline", "i1", 1.0)
            self._write_reward_run(run_root / "runs" / "candidate", "i1", 0.0)
            decision = self._make_decision(
                accepted=False,
                reason="candidate worse",
                baseline_run_id="baseline",
                candidate_run_id="candidate",
            )

            mod._write_round_ledger(
                run_root,
                1,
                decision,
                lambda run: run.reward if run.reward is not None else 0.0,
            )

            best_ever = json.loads(
                (run_root / "ahe" / "best_ever.json").read_text(encoding="utf-8")
            )

        self.assertEqual(best_ever["version"], "baseline-hash")
        self.assertEqual(best_ever["reward_mean"], 1.0)

    def test_ahe_recipe_runs_generic_dry_run_with_registered_swebench(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            train.write_text('{"instance_id": "sympy__sympy-1"}\n', encoding="utf-8")
            config = root / "ahe.yaml"
            config.write_text(
                f"""
run:
  id: temp-ahe
  output_root: {root / "out"}
  execute: false
  reset: false
  dotenv: .env
suite:
  name: swebench
surface:
  name: ahe_harness_surface
  editable_components: [everything]
  artifact_key: input/agent_package.json
  default: ahe_harness_package
instances:
  train:
    id: train
    path: {train}
execution:
  backend:
    name: local_docker
  store:
    name: local_dir
  parallel: 1
  max_turns: 3
model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
strategy:
  name: ahe_model
evolution:
  algorithm: simple
  rounds: 2
  criterion:
    name: promote_not_worse
    args:
      dim: reward
evaluation:
  baseline_heldout: false
  final_heldout: false
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: false
""".lstrip(),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "recipes/ahe/evolve.py",
                    "--config",
                    str(config),
                    "--run-id",
                    "override-ahe",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("run id: override-ahe", result.stdout)
        self.assertIn("suite: swebench", result.stdout)
        self.assertIn("surface: ahe_harness_surface", result.stdout)
        self.assertIn("editable components: everything", result.stdout)

    def test_ahe_recipe_default_config_runs_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "recipes/ahe/evolve.py",
                "--run-id",
                "default-ahe-smoke",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("run id: default-ahe-smoke", result.stdout)

    def _make_decision(
        self,
        *,
        accepted: bool,
        reason: str,
        baseline_run_id: str,
        candidate_run_id: str,
    ):
        from simple_agent_lab.evolution.types import Decision

        return Decision(
            id="d-000001",
            ts="2026-06-24T00:00:00Z",
            baseline={"hash": "baseline-hash", "scores": {"reward": 0.5}},
            candidate={"hash": "candidate-hash", "scores": {"reward": 0.5}},
            slice={"id": "train", "sha": "demo", "n": 2},
            accepted=accepted,
            reason=reason,
            runs={"baseline": baseline_run_id, "candidate": candidate_run_id},
            kind="ahe_harness",
        )

    def _write_reward_run(
        self, run_root: Path, instance_id: str, reward: float
    ) -> None:
        out_dir = run_root / instance_id / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps({"reward": reward}),
            encoding="utf-8",
        )


class DgmRecipeSmokeTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "evolve.py")

    def _write_dgm_config(self, root: Path) -> Path:
        train = root / "train.jsonl"
        test = root / "test.jsonl"
        train.write_text('{"instance_id": "train-1"}\n', encoding="utf-8")
        test.write_text('{"instance_id": "test-1"}\n', encoding="utf-8")
        config = root / "dgm.yaml"
        config.write_text(
            f"""
run:
  id: yaml-dgm
  output_root: {root / "out"}
  execute: false
  reset: false
  dotenv: .env.test
dataset:
  name: demo-dataset
  train_path: {train}
  test_path: {test}
execution:
  parallel: 3
  max_turns: 11
  wheelhouse: {root / "wheelhouse"}
  uv_binary: ""
model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  default_model: yaml-model
dgm:
  rounds: 4
  branches: 3
  meta_concurrency: 0
  parent_selection: score_child_prop
""".lstrip(),
            encoding="utf-8",
        )
        return config

    def test_loads_dgm_yaml_config(self):
        from recipes.dgm.config import load_dgm_config

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_dgm_config(Path(tmp))

            config = load_dgm_config(path)

        self.assertEqual(config.run.id, "yaml-dgm")
        self.assertEqual(config.dataset.name, "demo-dataset")
        self.assertEqual(config.execution.parallel, 3)
        self.assertEqual(config.dgm.branches, 3)

    def test_configure_args_uses_yaml_defaults_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_model = os.environ.pop("OPENAI_MODEL", None)
            self.addCleanup(
                lambda: (
                    os.environ.__setitem__("OPENAI_MODEL", old_model)
                    if old_model is not None
                    else os.environ.pop("OPENAI_MODEL", None)
                )
            )
            path = self._write_dgm_config(Path(tmp))
            ns = self.mod.build_parser().parse_args(
                [
                    "--config",
                    str(path),
                    "--run-id",
                    "override-dgm",
                    "--rounds",
                    "6",
                    "--parallel",
                    "4",
                ]
            )

            configured = self.mod.configure_args(ns)

        self.assertEqual(configured.run_id, "override-dgm")
        self.assertEqual(configured.rounds, 6)
        self.assertEqual(configured.branches, 3)
        self.assertEqual(configured.parallel, 4)
        self.assertEqual(configured.train_dataset.endswith("train.jsonl"), True)
        self.assertEqual(configured.model_name, "yaml-model")

    def test_configure_args_loads_config_dotenv_before_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_model = os.environ.pop("OPENAI_MODEL", None)
            self.addCleanup(
                lambda: (
                    os.environ.__setitem__("OPENAI_MODEL", old_model)
                    if old_model is not None
                    else os.environ.pop("OPENAI_MODEL", None)
                )
            )
            path = self._write_dgm_config(root)
            dotenv = root / ".env.test"
            dotenv.write_text("OPENAI_MODEL=dotenv-dgm-model\n", encoding="utf-8")
            text = path.read_text(encoding="utf-8")
            text = text.replace("  dotenv: .env.test\n", f"  dotenv: {dotenv}\n")
            path.write_text(text, encoding="utf-8")
            ns = self.mod.build_parser().parse_args(["--config", str(path)])

            configured = self.mod.configure_args(ns)

        self.assertEqual(configured.model_name, "dotenv-dgm-model")

    def test_parser_exposes_dgm_knobs(self):
        ns = self.mod.build_parser().parse_args(
            [
                "--parent-selection",
                "best",
                "--branches",
                "4",
            ]
        )
        self.assertEqual(ns.parent_selection, "best")
        self.assertEqual(ns.branches, 4)
        self.assertFalse(ns.execute)

    def test_parser_defaults_to_three_branch_dgm_recipe(self):
        ns = self.mod.configure_args(self.mod.build_parser().parse_args([]))

        self.assertEqual(ns.branches, 3)
        self.assertEqual(ns.parallel, 3)

    def test_pick_best_node_selects_highest_valid(self):
        from recipes.dgm.algorithm import archive

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
        ns = self.mod.configure_args(
            self.mod.build_parser().parse_args(
                [
                    "--rounds",
                    "6",
                    "--generations",
                    "9",
                    "--branches",
                    "2",
                ]
            )
        )
        rounds, branches, meta = self.mod.resolve_schedule(ns)
        self.assertEqual((rounds, branches, meta), (6, 2, 2))

    def test_resolve_schedule_falls_back_to_generations_then_default(self):
        ns_gen = self.mod.configure_args(
            self.mod.build_parser().parse_args(["--generations", "7"])
        )
        self.assertEqual(self.mod.resolve_schedule(ns_gen)[0], 7)
        ns_def = self.mod.configure_args(self.mod.build_parser().parse_args([]))
        rounds, branches, meta = self.mod.resolve_schedule(ns_def)
        self.assertEqual(rounds, 4)
        self.assertEqual(meta, branches)

    def test_default_config_runs_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "recipes/dgm/evolve.py",
                "--run-id",
                "default-dgm-smoke",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry run only", result.stdout)
        self.assertIn("run root:", result.stdout)

    def test_validate_schedule_capacity_rejects_more_branches_than_workers(self):
        with self.assertRaisesRegex(SystemExit, "--parallel.*--branches"):
            self.mod.validate_schedule_capacity(branches=4, global_workers=3)

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

    def test_run_workflow_rejects_unsafe_run_id_before_dataset_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.mod.build_parser().parse_args(
                [
                    "--run-id",
                    "../victim",
                    "--output-root",
                    str(root / "out"),
                    "--train-dataset",
                    str(root / "missing-train.jsonl"),
                    "--test-dataset",
                    str(root / "missing-test.jsonl"),
                    "--reset",
                ]
            )

            with self.assertRaisesRegex(ValueError, "unsafe run id"):
                self.mod.run_workflow(args)


class RecordHeldoutGenerationTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "evolve.py")
        from recipes.dgm import swebench as er

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

    def test_writes_generation_metrics_from_scoped_eval_results(self):
        from simple_agent_lab.trace.jsonl import write_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            layout = self.er.PerformanceLayout(Path(tmp), "run")
            layout.create()
            artifacts = self.er.official_artifacts(layout, "final")
            write_jsonl(artifacts.eval_results, self._eval_rows())
            record = self.mod.record_heldout_generation(
                layout,
                SimpleNamespace(hash="v1", parent="v0"),
                parent_selection="best",
                eval_results=artifacts.eval_results,
            )

        self.assertEqual(record["resolved"], 1)
        self.assertEqual(record["total"], 2)
        self.assertEqual(record["test_resolved_rate"], 0.5)

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
        report = _load(ROOT / "recipes" / "dgm" / "ops" / "report.py")
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
