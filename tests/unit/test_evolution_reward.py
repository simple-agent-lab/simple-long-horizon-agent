from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.components.reward import cost_tokens, result_key
from simple_agent_lab.evolution.types import Run


class RewardTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _run(self, *, result: dict, events: list[dict] | None = None) -> Run:
        d = self.tmp / "r" / "i1"
        (d / "out").mkdir(parents=True)
        (d / "out" / "result.json").write_text(json.dumps(result))
        if events is not None:
            (d / "out" / "trajectory.jsonl").write_text(json.dumps({"events": events}))
        return Run(d)

    def test_result_key_reads_reward(self) -> None:
        self.assertEqual(result_key(self._run(result={"reward": 0.7})), 0.7)

    def test_result_key_crash_is_zero(self) -> None:
        crashed = Run(self.tmp / "r" / "missing")
        self.assertEqual(result_key(crashed), 0.0)

    def test_cost_tokens_sums_usage(self) -> None:
        run = self._run(
            result={"reward": 1.0},
            events=[
                {"usage": {"input_tokens": 10, "output_tokens": 5}},
                {"usage": {"input_tokens": 20, "output_tokens": 0}},
                {"kind": "other"},
            ],
        )
        self.assertEqual(cost_tokens(run), 35.0)


if __name__ == "__main__":
    unittest.main()
