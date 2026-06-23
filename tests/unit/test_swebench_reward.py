import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simple_agent_lab.evolution.types import Run

from recipes import swebench_reward


class SwebenchRewardTest(unittest.TestCase):
    def _run(self, result):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name) / "run" / "case-1"
        out = run_dir / "out"
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
        return Run(run_dir), out / "result.json"

    def test_reward_from_result_prefers_existing_verdict_fields(self):
        self.assertEqual(swebench_reward.reward_from_result({"resolved": True}), 1.0)
        self.assertEqual(swebench_reward.reward_from_result({"resolved": False}), 0.0)
        self.assertEqual(swebench_reward.reward_from_result({"score": 0.25}), 0.25)
        self.assertEqual(swebench_reward.reward_from_result({"reward": 0.75}), 0.75)
        self.assertEqual(swebench_reward.reward_from_result({}), 0.0)

    def test_reward_from_result_penalizes_agent_package_fallback(self):
        self.assertEqual(
            swebench_reward.reward_from_result(
                {"resolved": True, "agent_package": {"used_fallback": True}}
            ),
            -1.0,
        )

    def test_apply_eval_score_updates_result_json(self):
        run, result_path = self._run({"model_patch": "diff"})
        swebench_reward.apply_eval_score(
            run,
            {
                "passed": True,
                "score": 1.0,
                "reason": "resolved",
                "metrics": {"resolved": True, "status": "resolved"},
            },
        )

        result = json.loads(result_path.read_text())
        self.assertTrue(result["resolved"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["status"], "resolved")

    def test_make_reuse_reward_enriches_raw_eval_log_result(self):
        run, result_path = self._run({"model_patch": "diff", "eval_log": "log"})
        reward = swebench_reward.make_reuse_reward(
            instances=[{"instance_id": "case-1", "repo": "owner/repo"}],
            dataset_name="dataset",
            model_name="model",
        )

        with patch.object(
            swebench_reward,
            "reuse_eval_row",
            return_value={
                "passed": True,
                "score": 1.0,
                "reason": "resolved",
                "metrics": {"resolved": True, "status": "resolved"},
            },
        ) as reuse:
            self.assertEqual(reward(run), 1.0)

        reuse.assert_called_once()
        result = json.loads(result_path.read_text())
        self.assertTrue(result["resolved"])
        self.assertEqual(result["reward"], 1.0)

    def test_make_reuse_reward_writes_diagnostic_on_scoring_error(self):
        run, result_path = self._run({"model_patch": "diff", "eval_log": "log"})
        reward = swebench_reward.make_reuse_reward(
            instances=[{"instance_id": "case-1"}],
            dataset_name="dataset",
            model_name="model",
        )

        with patch.object(swebench_reward, "reuse_eval_row", side_effect=RuntimeError("boom")):
            self.assertEqual(reward(run), 0.0)

        result = json.loads(result_path.read_text())
        self.assertFalse(result["resolved"])
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("RuntimeError: boom", result["scoring_error"])

    def test_make_reuse_reward_does_not_rescore_existing_reward(self):
        run, _ = self._run({"reward": 0.5, "eval_log": "log"})
        reward = swebench_reward.make_reuse_reward(
            instances=[{"instance_id": "case-1"}],
            dataset_name="dataset",
            model_name="model",
        )

        with patch.object(swebench_reward, "reuse_eval_row") as reuse:
            self.assertEqual(reward(run), 0.5)

        reuse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
