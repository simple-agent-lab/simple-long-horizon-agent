from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recipes.ahe.ledger import (
    ahe_root,
    append_history,
    evaluate_manifest_predictions,
    read_json,
    round_dir,
    update_best_ever,
    update_task_history,
    write_json,
)


class AheLedgerTest(unittest.TestCase):
    def test_round_dir_creates_padded_round_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)

            path = round_dir(run_root, 1)
            self.assertEqual(path, ahe_root(run_root) / "rounds" / "round_001")
            self.assertTrue(path.is_dir())

    def test_write_and_read_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "data.json"

            write_json(path, {"b": 2, "a": {"c": 3}})
            data = read_json(path, default={})
            self.assertEqual(data, {"a": {"c": 3}, "b": 2})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "a": {\n    "c": 3\n  },\n  "b": 2\n}\n',
            )

    def test_update_task_history_records_scores_and_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)

            history = update_task_history(
                run_root,
                3,
                {"i1": {"reward": 0}, "i2": {"reward": 1}},
            )

            stored = json.loads(
                (ahe_root(run_root) / "task_history.json").read_text(encoding="utf-8")
            )

        self.assertEqual(history, stored)
        self.assertEqual(history["i1"][0]["round"], 3)
        self.assertEqual(history["i1"][0]["scores"], {"reward": 0})
        self.assertFalse(history["i1"][0]["passed"])
        self.assertTrue(history["i2"][0]["passed"])

    def test_evaluate_manifest_predictions_mixed(self) -> None:
        manifest = {
            "round": 1,
            "changes": [
                {
                    "id": "chg-1",
                    "predicted_fixes": ["i1"],
                    "risk_tasks": ["i2"],
                }
            ],
        }
        baseline_scores = {"i1": {"reward": 0}, "i2": {"reward": 1}}
        candidate_scores = {"i1": {"reward": 1}, "i2": {"reward": 0}}

        result = evaluate_manifest_predictions(
            manifest, baseline_scores, candidate_scores
        )

        self.assertEqual(result["round"], 1)
        self.assertEqual(result["fixed_tasks"], ["i1"])
        self.assertEqual(result["regressed_tasks"], ["i2"])
        change = result["change_evaluations"][0]
        self.assertEqual(change["expected_fixes_verified"], ["i1"])
        self.assertEqual(change["false_predictions"], [])
        self.assertEqual(change["regressions_observed"], ["i2"])
        self.assertEqual(change["unexpected_fixes"], [])
        self.assertEqual(change["verdict"], "mixed")

    def test_evaluate_manifest_predictions_ineffective_and_unexpected_fix(self) -> None:
        manifest = {
            "round": 2,
            "changes": [
                {
                    "id": "chg-2",
                    "predicted_fixes": ["i3"],
                    "risk_tasks": ["i2"],
                }
            ],
        }
        baseline_scores = {"i1": {"reward": 0}, "i2": {"reward": 1}}
        candidate_scores = {"i1": {"reward": 1}, "i2": {"reward": 1}}

        result = evaluate_manifest_predictions(
            manifest, baseline_scores, candidate_scores
        )

        change = result["change_evaluations"][0]
        self.assertEqual(change["expected_fixes_verified"], [])
        self.assertEqual(change["false_predictions"], ["i3"])
        self.assertEqual(change["regressions_observed"], [])
        self.assertEqual(change["unexpected_fixes"], ["i1"])
        self.assertEqual(change["verdict"], "ineffective")

    def test_update_best_ever_keeps_highest_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)

            first = update_best_ever(
                run_root,
                1,
                "v1",
                {"i1": {"reward": 0}, "i2": {"reward": 1}},
            )
            second = update_best_ever(
                run_root,
                2,
                "v2",
                {"i1": {"reward": 0}, "i2": {"reward": 0}},
            )
            stored = json.loads(
                (ahe_root(run_root) / "best_ever.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            first, {"round": 1, "version": "v1", "reward_mean": 0.5, "total": 2}
        )
        self.assertEqual(second, first)
        self.assertEqual(stored, first)

    def test_append_history_creates_file_and_appends_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)

            append_history(run_root, "first entry")
            append_history(run_root, "second entry\n")
            text = (ahe_root(run_root) / "history.md").read_text(encoding="utf-8")

        self.assertEqual(text, "first entry\nsecond entry\n")


if __name__ == "__main__":
    unittest.main()
