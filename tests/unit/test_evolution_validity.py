import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.components.criterion import valid_when
from simple_agent_lab.evolution.kernel import log


def _scores(reward):
    return {"i1": {"reward": reward}}


class ValidWhenTest(unittest.TestCase):
    def test_admits_any_gradable_child(self):
        verdict = valid_when()(_scores(0.0), _scores(0.0))
        self.assertIs(verdict.accepted, True)
        self.assertEqual(verdict.deltas["valid_parent"], 1.0)

    def test_rejects_when_no_runs(self):
        verdict = valid_when()({}, {})
        self.assertIs(verdict.accepted, False)
        self.assertEqual(verdict.deltas["valid_parent"], 0.0)

    def test_records_reward_delta(self):
        verdict = valid_when()(_scores(0.5), _scores(0.4))
        self.assertIs(verdict.accepted, True)  # worse, but valid -> still admitted
        self.assertAlmostEqual(verdict.deltas["reward"], -0.1)


class LogValidParentTest(unittest.TestCase):
    def test_log_append_preserves_valid_parent_on_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            log.append(
                workspace,
                baseline={"hash": "p", "scores": {"reward": 0.5}},
                candidate={
                    "hash": "c",
                    "parent": "p",
                    "scores": {"reward": 0.4},
                    "valid_parent": True,
                },
                slice_=type("S", (), {"id": "s", "sha": "abc", "n": 1})(),
                verdict=type(
                    "V", (), {"accepted": True, "reason": "ok", "deltas": {}}
                )(),
                kind="code",
                runs={"baseline": "rb", "candidate": "rc"},
            )
            rows = log.read(workspace)
            self.assertIs(rows[-1].candidate["valid_parent"], True)


if __name__ == "__main__":
    unittest.main()
