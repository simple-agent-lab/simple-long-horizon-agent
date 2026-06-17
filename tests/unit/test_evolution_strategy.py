import unittest
from pathlib import Path
import tempfile

from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.components.strategy import (
    model_program_strategy,
    parse_model_json,
    safe_prefix_edits,
)
from simple_agent_lab.evolution.types import Context


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def fake_complete(_request):
    return FakeResponse(
        '{"note": "n", "evidence": ["e"], '
        '"edits": {"agent/agent_program.py": "X = 1\\n", "../escape.py": "bad"}}'
    )


class StrategyComponentTest(unittest.TestCase):
    def test_parse_model_json_handles_fenced(self):
        self.assertEqual(parse_model_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_safe_prefix_edits_rejects_escape_and_bad_python(self):
        edits, rejected = safe_prefix_edits(
            {"agent/ok.py": "x = 1\n", "agent/bad.py": "def (", "../e.py": "x"},
            prefix="agent/",
        )
        self.assertIn("agent/ok.py", edits)
        self.assertIn("agent/bad.py", rejected)
        self.assertIn("../e.py", rejected)

    def test_safe_prefix_edits_keeps_none_tombstone(self):
        edits, rejected = safe_prefix_edits(
            {"agent/old.py": None, "../e.py": None}, prefix="agent/"
        )
        self.assertIsNone(edits["agent/old.py"])
        self.assertIn("../e.py", rejected)

    def test_strategy_returns_prefixed_edits_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            strat = model_program_strategy(
                provider=object(), prefix="agent/", complete_fn=fake_complete
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )
            proposal = strat(ctx)
            self.assertIn("agent/agent_program.py", proposal.edits)
            self.assertTrue(
                any("discarded-disallowed-path" in e for e in proposal.evidence)
            )


if __name__ == "__main__":
    unittest.main()
