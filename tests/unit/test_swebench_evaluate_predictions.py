from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evals.swebench import evaluate_predictions


class SwebenchEvaluatePredictionsTest(unittest.TestCase):
    def test_default_official_and_pro_harness_paths_live_under_suite_output(
        self,
    ) -> None:
        self.assertEqual(
            evaluate_predictions.DEFAULT_OFFICIAL_OUTPUT_DIR,
            evaluate_predictions.ROOT / "evals/out/swebench_official",
        )
        self.assertEqual(
            evaluate_predictions.DEFAULT_MULTILINGUAL_OFFICIAL_OUTPUT_DIR,
            evaluate_predictions.ROOT / "evals/out/swebench_multilingual_official",
        )
        self.assertEqual(
            evaluate_predictions.DEFAULT_PRO_EVAL_SCRIPT,
            evaluate_predictions.ROOT
            / "evals/out/swebench_pro/official_harness/swe_bench_pro_eval.py",
        )
        self.assertEqual(
            evaluate_predictions.DEFAULT_PRO_SCRIPTS_DIR,
            evaluate_predictions.ROOT
            / "evals/out/swebench_pro/official_harness/run_scripts",
        )

    def test_pro_mode_defaults_to_pro_output_paths(self) -> None:
        args = evaluate_predictions.parse_args(["--pro"])

        self.assertEqual(
            args.predictions,
            str(
                evaluate_predictions.ROOT
                / "evals/out/swebench_pro/swebench_pro_predictions.jsonl"
            ),
        )
        # Eval results are now bound to the run that produced the predictions:
        # <predictions_dir>/<run_id>/eval_results.jsonl (run_id = predictions
        # stem minus the _predictions suffix).
        self.assertEqual(
            args.jsonl,
            str(
                evaluate_predictions.ROOT
                / "evals/out/swebench_pro/swebench_pro/eval_results.jsonl"
            ),
        )
        self.assertEqual(
            args.official_output_dir,
            str(evaluate_predictions.DEFAULT_PRO_OFFICIAL_OUTPUT_DIR),
        )

    def test_multilingual_mode_defaults_to_multilingual_output_paths(self) -> None:
        args = evaluate_predictions.parse_args(["--multilingual"])

        self.assertEqual(
            args.predictions,
            str(
                evaluate_predictions.ROOT
                / "evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl"
            ),
        )
        self.assertEqual(
            args.jsonl,
            str(
                evaluate_predictions.ROOT
                / "evals/out/swebench_multilingual/swebench_multilingual/eval_results.jsonl"
            ),
        )
        self.assertEqual(
            args.official_output_dir,
            str(evaluate_predictions.DEFAULT_MULTILINGUAL_OFFICIAL_OUTPUT_DIR),
        )
        self.assertEqual(args.dataset_name, "SWE-bench/SWE-bench_Multilingual")

    def test_eval_results_default_binds_to_run_dir_of_predictions(self) -> None:
        # A run-specific predictions file (<run_id>_predictions.jsonl) puts the
        # verdicts inside that run's directory, beside its trajectories.
        args = evaluate_predictions.parse_args(
            [
                "--multilingual",
                "--predictions",
                "evals/out/swebench_multilingual/multilingual-3-20260609-164525_predictions.jsonl",
            ]
        )
        self.assertEqual(
            args.jsonl,
            "evals/out/swebench_multilingual/multilingual-3-20260609-164525/eval_results.jsonl",
        )

    def test_eval_results_default_strips_fixed_suffix(self) -> None:
        args = evaluate_predictions.parse_args(
            [
                "--predictions",
                "evals/out/swebench/my-run_predictions.fixed.jsonl",
            ]
        )
        self.assertEqual(
            args.jsonl,
            "evals/out/swebench/my-run/eval_results.jsonl",
        )

    def test_official_paths_default_under_run_specific_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                official_output_dir=str(Path(tmp) / "official"),
                run_id="responses:astropy/12907",
                report_dir=None,
            )

            run_dir = evaluate_predictions.official_run_dir(args)
            report_dir = evaluate_predictions.official_report_dir(args)

        self.assertEqual(run_dir.name, "responses_astropy_12907")
        self.assertEqual(report_dir, run_dir / "reports")

    def test_prediction_path_for_harness_preserves_gold_sentinel(self) -> None:
        self.assertEqual(
            evaluate_predictions.prediction_path_for_harness("gold"), "gold"
        )

    def test_prediction_path_for_harness_resolves_relative_paths(self) -> None:
        resolved = evaluate_predictions.prediction_path_for_harness(
            "evals/out/prediction.jsonl"
        )

        self.assertEqual(
            resolved,
            str(evaluate_predictions.ROOT / "evals/out/prediction.jsonl"),
        )

    def test_run_official_harness_uses_run_dir_as_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                dataset_name="princeton-nlp/SWE-bench_Verified",
                split="test",
                predictions="gold",
                max_workers=1,
                run_id="validate-gold",
                official_output_dir=str(Path(tmp) / "official"),
                report_dir=None,
                instance_ids=["sympy__sympy-23824"],
                cache_level="",
                clean=False,
                timeout=None,
            )
            with (
                mock.patch.object(
                    evaluate_predictions.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                mock.patch.object(evaluate_predictions.subprocess, "run") as run,
            ):
                evaluate_predictions.run_official_harness(args)

            command = run.call_args.args[0]
            cwd = run.call_args.kwargs["cwd"]

        self.assertEqual(cwd, evaluate_predictions.official_run_dir(args))
        self.assertIn("--predictions_path", command)
        self.assertIn("gold", command)
        self.assertIn(str(evaluate_predictions.official_report_dir(args)), command)

    def test_load_predictions_accepts_pro_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
                            "prefix": "simple-agent-lab-pro",
                            "patch": "diff --git a/api.js b/api.js\n",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = evaluate_predictions.load_predictions(path)

        self.assertEqual(records[0]["prefix"], "simple-agent-lab-pro")
        self.assertEqual(records[0]["patch"], "diff --git a/api.js b/api.js\n")

    def test_load_predictions_streams_jsonl_without_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "instance_id": "sympy__sympy-23824",
                        "model_name_or_path": "model",
                        "model_patch": "diff --git a/a b/a\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("JSONL should be read from a file handle"),
            ):
                records = evaluate_predictions.load_predictions(path)

        self.assertEqual(records[0]["instance_id"], "sympy__sympy-23824")

    def test_collect_predictions_uses_collected_model_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "run-1" / "instance_one"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "out").mkdir()
            (run_dir / "input" / "instance.json").write_text(
                json.dumps({"instance_id": "instance_one"}),
                encoding="utf-8",
            )
            (run_dir / "out" / "result.json").write_text(
                json.dumps({"model_patch": "diff --git a/app.py b/app.py\n"}),
                encoding="utf-8",
            )

            predictions = evaluate_predictions.predictions_from_run_dirs(
                Path(tmp) / "runs",
                run_id="run-1",
                model_name="model",
                dataset_name="ScaleAI/SWE-bench_Pro",
            )

        self.assertEqual(predictions[0]["prefix"], "model")
        self.assertEqual(predictions[0]["patch"], "diff --git a/app.py b/app.py\n")

    def test_collect_predictions_emits_empty_patch_for_expected_missing_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            run_dir = run_root / "run-1" / "instance_one"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "out").mkdir()
            (run_dir / "input" / "instance.json").write_text(
                json.dumps({"instance_id": "instance_one"}),
                encoding="utf-8",
            )
            (run_dir / "out" / "result.json").write_text(
                json.dumps({"model_patch": "diff --git a/a b/a\n"}),
                encoding="utf-8",
            )

            predictions = evaluate_predictions.predictions_from_run_dirs(
                run_root,
                run_id="run-1",
                model_name="model",
                dataset_name="ScaleAI/SWE-bench_Pro",
                expected_instance_ids=("instance_one", "instance_two"),
            )

        self.assertEqual(
            [prediction["instance_id"] for prediction in predictions],
            ["instance_one", "instance_two"],
        )
        self.assertEqual(predictions[1]["patch"], "")

    def test_collect_predictions_rejects_result_outside_expected_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            stale_dir = run_root / "run-1" / "stale_instance"
            (stale_dir / "input").mkdir(parents=True)
            (stale_dir / "out").mkdir()
            (stale_dir / "input" / "instance.json").write_text(
                json.dumps({"instance_id": "stale_instance"}),
                encoding="utf-8",
            )
            (stale_dir / "out" / "result.json").write_text(
                json.dumps({"model_patch": "stale diff"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unexpected instance id"):
                evaluate_predictions.predictions_from_run_dirs(
                    run_root,
                    run_id="run-1",
                    expected_instance_ids=("planned_instance",),
                )

    def test_collect_predictions_rejects_duplicate_instance_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            for dirname in ("one", "duplicate"):
                run_dir = run_root / "run-1" / dirname
                (run_dir / "input").mkdir(parents=True)
                (run_dir / "out").mkdir()
                (run_dir / "input" / "instance.json").write_text(
                    json.dumps({"instance_id": "instance_one"}),
                    encoding="utf-8",
                )
                (run_dir / "out" / "result.json").write_text(
                    json.dumps({"model_patch": "diff"}),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "Duplicate result"):
                evaluate_predictions.predictions_from_run_dirs(
                    run_root,
                    run_id="run-1",
                    expected_instance_ids=("instance_one",),
                )

    def test_results_from_summary_accepts_pro_eval_results_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_results.json"
            path.write_text(
                json.dumps(
                    {
                        "instance_NodeBB__NodeBB-abc-vnan": True,
                        "instance_NodeBB__NodeBB-def-vnan": False,
                    }
                ),
                encoding="utf-8",
            )

            results = evaluate_predictions.results_from_summary(path)

        self.assertTrue(results["instance_NodeBB__NodeBB-abc-vnan"]["resolved"])
        self.assertEqual(
            results["instance_NodeBB__NodeBB-abc-vnan"]["status"],
            "resolved",
        )
        self.assertFalse(results["instance_NodeBB__NodeBB-def-vnan"]["resolved"])
        self.assertEqual(
            results["instance_NodeBB__NodeBB-def-vnan"]["status"],
            "unresolved",
        )

    def test_pro_eval_result_counts_patch_and_prefix_fields(self) -> None:
        result = evaluate_predictions.eval_result_from_official(
            {
                "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
                "prefix": "simple-agent-lab-pro",
                "patch": "diff --git a/api.js b/api.js\n",
            },
            {
                "resolved": True,
                "status": "resolved",
                "report_source": "eval_results.json",
            },
        )

        self.assertEqual(result.scorer, "swebench_pro.official_harness.v1")
        self.assertEqual(result.metrics["model_name_or_path"], "simple-agent-lab-pro")
        self.assertEqual(result.metrics["patch_chars"], 29)
        self.assertIsNotNone(result.meta)
        assert result.meta is not None
        self.assertEqual(result.meta["suite"], "swebench_pro")

    def test_multilingual_eval_result_uses_distinct_suite_metadata(self) -> None:
        result = evaluate_predictions.eval_result_from_official(
            {
                "instance_id": "kotlin__repo-123",
                "model_name_or_path": "simple-agent-lab-multilingual",
                "model_patch": "diff --git a/src/App.kt b/src/App.kt\n",
            },
            {
                "resolved": False,
                "status": "unresolved",
                "report_source": "report.json",
            },
            suite="swebench_multilingual",
        )

        self.assertEqual(result.scorer, "swebench_multilingual.official_harness.v1")
        self.assertEqual(
            result.metrics["model_name_or_path"],
            "simple-agent-lab-multilingual",
        )
        self.assertIsNotNone(result.meta)
        assert result.meta is not None
        self.assertEqual(result.meta["suite"], "swebench_multilingual")

    def test_run_official_pro_harness_accepts_jsonl_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "instance_id": "instance_one",
                        "prefix": "model",
                        "patch": "diff --git a/a b/a\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instances = root / "instances.jsonl"
            instances.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "instance_id": "instance_one",
                                "fail_to_pass": ["test_a"],
                                "pass_to_pass": ["test_b"],
                            }
                        ),
                        json.dumps(
                            {
                                "instance_id": "instance_two",
                                "fail_to_pass": ["test_c"],
                                "pass_to_pass": ["test_d"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            script = root / "swe_bench_pro_eval.py"
            script.write_text("# fake evaluator\n", encoding="utf-8")
            report_dir = root / "report"
            stale = report_dir / "official" / "instance_stale" / "workspace"
            stale.mkdir(parents=True)
            (stale / "output.json").write_text('{"stale": true}')

            def fake_run(command, cwd, check):
                del command, cwd, check
                self.assertFalse(stale.exists())
                official = report_dir / "official"
                official.mkdir(parents=True)
                (official / "eval_results.json").write_text(
                    json.dumps({"instance_one": True}), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0)

            args = Namespace(
                instances=str(instances),
                pro_eval_script=str(script),
                predictions=str(predictions),
                report_dir=str(report_dir),
                instance_ids=[],
                dockerhub_username="jefzda",
                scripts_dir=str(root / "run_scripts"),
            )

            with (
                mock.patch.object(
                    evaluate_predictions.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
                mock.patch.object(evaluate_predictions, "_patch_pro_evaluator"),
            ):
                evaluate_predictions.run_official_pro_harness(args)

            prepared = (report_dir / "instances_for_official.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(len(prepared.strip().splitlines()), 2)
        self.assertIn('"instance_id": "instance_one"', prepared)
        self.assertIn('"instance_id": "instance_two"', prepared)

    def test_load_instance_records_streams_jsonl_without_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "instance_id": "instance_one",
                        "fail_to_pass": ["test_a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("JSONL should be read from a file handle"),
            ):
                records = evaluate_predictions._load_instance_records(path)

        self.assertEqual(records[0]["instance_id"], "instance_one")

    def test_run_official_pro_harness_fails_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "instance_id": "instance_one",
                        "prefix": "model",
                        "patch": "diff --git a/a b/a\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instances = root / "instances.jsonl"
            instances.write_text(
                json.dumps({"instance_id": "instance_one"}) + "\n",
                encoding="utf-8",
            )
            script = root / "swe_bench_pro_eval.py"
            script.write_text("# fake evaluator\n", encoding="utf-8")
            report_dir = root / "report"
            stale = report_dir / "official"
            stale.mkdir(parents=True)
            (stale / "eval_results.json").write_text(
                json.dumps({"instance_one": True}), encoding="utf-8"
            )

            args = Namespace(
                instances=str(instances),
                pro_eval_script=str(script),
                predictions=str(predictions),
                report_dir=str(report_dir),
                instance_ids=[],
                dockerhub_username="jefzda",
                scripts_dir=str(root / "run_scripts"),
            )

            with (
                mock.patch.object(
                    evaluate_predictions.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=2),
                ),
                mock.patch.object(evaluate_predictions, "_patch_pro_evaluator"),
            ):
                with self.assertRaisesRegex(SystemExit, "exited with 2"):
                    evaluate_predictions.run_official_pro_harness(args)

    def test_patch_pro_evaluator_enforces_apply_check_and_status(self) -> None:
        source = '''
def create_entryscript(sample):
    base_commit = sample["base_commit"]
    before_repo_set_cmd = "prepare tests"
    entry_script = f"""
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
"""
    return entry_script

client = docker.from_env()
def pull():
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                return None
'''
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "swe_bench_pro_eval.py"
            script.write_text(source, encoding="utf-8")

            evaluate_predictions._patch_pro_evaluator(script)
            patched = script.read_text(encoding="utf-8")

        self.assertIn("git apply --check", patched)
        self.assertIn("patch_apply_status.json", patched)
        self.assertIn('"success": true', patched)
        self.assertIn("docker.from_env(timeout=600)", patched)
        self.assertIn("Using locally cached Docker image", patched)

    def test_patch_pro_evaluator_rejects_unknown_entryscript_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "swe_bench_pro_eval.py"
            script.write_text("# unexpected upstream shape\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "entryscript"):
                evaluate_predictions._patch_pro_evaluator(script)

    def test_merge_pro_apply_status_forces_failed_patch_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            official = Path(tmp) / "official"
            workspace = official / "instance_one" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "patch_apply_status.json").write_text(
                json.dumps({"success": False, "stage": "check"}),
                encoding="utf-8",
            )
            (workspace / "patch_apply.stderr").write_text(
                "error: corrupt patch",
                encoding="utf-8",
            )
            results = {
                "instance_one": {
                    "resolved": True,
                    "status": "resolved",
                }
            }

            evaluate_predictions.merge_pro_apply_statuses(results, official)

        self.assertFalse(results["instance_one"]["resolved"])
        self.assertEqual(results["instance_one"]["status"], "patch_apply_failed")
        self.assertFalse(results["instance_one"]["patch_successfully_applied"])
        self.assertEqual(results["instance_one"]["patch_apply_stage"], "check")
        self.assertIn("corrupt patch", results["instance_one"]["patch_apply_stderr"])


def _instance(instance_id: str) -> dict[str, object]:
    return {"instance_id": instance_id, "repo": "acme/widget"}


def _separate_row(instance_id: str, *, resolved: bool) -> dict[str, object]:
    """An official ("separate") row, as `--run-official` would normalize it."""

    prediction = {"instance_id": instance_id, "model_name_or_path": "stub"}
    official = {
        "resolved": resolved,
        "status": "resolved" if resolved else "unresolved",
    }
    return evaluate_predictions.eval_result_record(
        evaluate_predictions.eval_result_from_official(prediction, official)
    )


class ReuseEvalRowTest(unittest.TestCase):
    """In-environment ("reuse") scoring helper, no Docker (ADR collapse-scorer-seam-into-run-primitive).

    `reuse_eval_row` grades what the container-half ``evaluate`` hook merged into
    ``result.json``. The no-Docker branches (an explicit verdict, a missing
    verdict) route through the same `eval_result_from_official` mapping as the
    official path, so the rows are interchangeable.
    """

    def test_explicit_verdict_maps_to_row(self) -> None:
        row = evaluate_predictions.reuse_eval_row(
            _instance("sympy__sympy-1"),
            {"model_patch": "diff", "resolved": True, "status": "resolved"},
        )
        self.assertEqual(row["metrics"]["instance_id"], "sympy__sympy-1")
        self.assertTrue(row["passed"])
        self.assertEqual(row["score"], 1.0)

    def test_unresolved_verdict_fails(self) -> None:
        row = evaluate_predictions.reuse_eval_row(
            _instance("sympy__sympy-2"),
            {"model_patch": "diff", "resolved": False},
        )
        self.assertFalse(row["passed"])

    def test_multilingual_verdict_keeps_multilingual_suite_metadata(self) -> None:
        row = evaluate_predictions.reuse_eval_row(
            _instance("kotlin__repo-123"),
            {"model_patch": "diff", "resolved": True},
            dataset_name="SWE-bench/SWE-bench_Multilingual",
        )
        self.assertEqual(row["scorer"], "swebench_multilingual.official_harness.v1")
        self.assertEqual(row["meta"]["suite"], "swebench_multilingual")

    def test_missing_verdict_is_unresolved_diagnostic(self) -> None:
        row = evaluate_predictions.reuse_eval_row(
            _instance("sympy__sympy-3"), {"model_patch": "diff"}
        )
        self.assertFalse(row["passed"])
        self.assertEqual(row["metrics"]["status"], "no_reuse_verdict")


class ParityGateTest(unittest.TestCase):
    """The parity gate (hard requirement, ADR collapse-scorer-seam-into-run-primitive): reuse must match official."""

    def test_parity_holds_when_reuse_matches_official(self) -> None:
        separate = [
            _separate_row("sympy__sympy-1", resolved=True),
            _separate_row("sympy__sympy-2", resolved=False),
        ]
        reuse = [
            evaluate_predictions.reuse_eval_row(
                _instance("sympy__sympy-1"), {"resolved": True}
            ),
            evaluate_predictions.reuse_eval_row(
                _instance("sympy__sympy-2"), {"resolved": False}
            ),
        ]
        self.assertEqual(evaluate_predictions.parity_mismatches(separate, reuse), [])

    def test_parity_flags_disagreement(self) -> None:
        separate = [_separate_row("sympy__sympy-1", resolved=True)]  # harness: pass
        reuse = [
            evaluate_predictions.reuse_eval_row(
                _instance("sympy__sympy-1"), {"resolved": False}
            )
        ]
        mismatches = evaluate_predictions.parity_mismatches(separate, reuse)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["instance_id"], "sympy__sympy-1")
        self.assertTrue(mismatches[0]["separate_passed"])
        self.assertFalse(mismatches[0]["reuse_passed"])


if __name__ == "__main__":
    unittest.main()
