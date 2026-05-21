from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from evals.swebench import evaluate_predictions


class SwebenchEvaluatePredictionsTest(unittest.TestCase):
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
                dataset_name="princeton-nlp/SWE-bench_Lite",
                split="test",
                predictions="gold",
                max_workers=1,
                run_id="validate-gold",
                official_output_dir=str(Path(tmp) / "official"),
                report_dir=None,
                instance_ids=["sympy__sympy-20590"],
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


if __name__ == "__main__":
    unittest.main()
