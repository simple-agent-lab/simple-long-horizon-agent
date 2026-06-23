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


class SequenceComplete:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, _request):
        text = self.texts[self.calls]
        self.calls += 1
        return FakeResponse(text)


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
            self.assertNotIn("../escape.py", proposal.edits)
            self.assertTrue(
                any("discarded-disallowed-path" in e for e in proposal.evidence)
            )

    def test_strategy_uses_surface_validation(self):
        from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
        from simple_agent_lab.evolution.surface import python_agent_surface

        surface = python_agent_surface(
            default_files={"agent_program.py": "def build_agent(**kwargs): pass\n"},
            artifact_key=AGENT_PACKAGE_KEY,
        )

        def complete_with_prompt_capture(request):
            self.assertIn(
                "Editable surface: Python agent package", request.system_prompt
            )
            self.assertIn("prompts", request.system_prompt)
            return FakeResponse(
                '{"note": "n", "evidence": [], '
                '"edits": {"agent/prompts.py": "SYSTEM_PROMPT = \\"x\\"\\n", '
                '"agent/tool_policy.py": "MAX_RETRIES = 3\\n"}}'
            )

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(ws, rollout=lambda v, s: [], seed=surface.seed_files())
            strat = model_program_strategy(
                provider=object(),
                surface=surface,
                editable_components=("prompts",),
                system_prompt="You are editing.\n",
                complete_fn=complete_with_prompt_capture,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )

            proposal = strat(ctx)

        self.assertIn("agent/prompts.py", proposal.edits)
        self.assertNotIn("agent/tool_policy.py", proposal.edits)
        self.assertTrue(
            any(
                "discarded-disallowed-path:agent/tool_policy.py" == e
                for e in proposal.evidence
            )
        )

    def test_strategy_declines_when_surface_edits_are_not_a_mapping(self):
        from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
        from simple_agent_lab.evolution.surface import python_agent_surface

        surface = python_agent_surface(
            default_files={"agent_program.py": "def build_agent(**kwargs): pass\n"},
            artifact_key=AGENT_PACKAGE_KEY,
        )
        response_texts = (
            '{"note": "n", "evidence": [], "edits": []}',
            '{"note": "n", "evidence": [], "edits": null}',
        )

        for response_text in response_texts:
            with self.subTest(response_text=response_text):
                with tempfile.TemporaryDirectory() as tmp:
                    ws = Path(tmp) / "ws"
                    exp = Experiment(
                        ws, rollout=lambda v, s: [], seed=surface.seed_files()
                    )
                    strat = model_program_strategy(
                        provider=object(),
                        surface=surface,
                        editable_components=("prompts",),
                        complete_fn=lambda _request: FakeResponse(response_text),
                    )
                    ctx = Context(
                        runs=(),
                        current=exp.current(),
                        workspace=ws,
                        decisions=(),
                        reward=lambda r: 0.0,
                    )

                    proposal = strat(ctx)

                self.assertIsNone(proposal)

    def test_strategy_declines_when_all_edits_are_rejected(self):
        from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
        from simple_agent_lab.evolution.surface import python_agent_surface

        surface = python_agent_surface(
            default_files={"agent_program.py": "def build_agent(**kwargs): pass\n"},
            artifact_key=AGENT_PACKAGE_KEY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(ws, rollout=lambda v, s: [], seed=surface.seed_files())
            strat = model_program_strategy(
                provider=object(),
                surface=surface,
                editable_components=("prompts",),
                complete_fn=lambda _request: FakeResponse(
                    '{"note": "n", "evidence": [], '
                    '"edits": {"agent/tool_policy.py": "MAX_RETRIES = 3\\n"}}'
                ),
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )

            proposal = strat(ctx)

        self.assertIsNone(proposal)

    def test_strategy_retries_invalid_json_before_returning_proposal(self):
        complete = SequenceComplete(
            "",
            '{"note": "n", "evidence": [], '
            '"edits": {"agent/agent_program.py": "X = 2\\n"}}',
        )
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                complete_fn=complete,
                log_fn=logs.append,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )

            proposal = strat(ctx)

            self.assertIsNotNone(proposal)
            self.assertEqual(complete.calls, 2)
            self.assertEqual(len(logs), 1)
            assert proposal is not None
            self.assertEqual(proposal.edits["agent/agent_program.py"], "X = 2\n")

    def test_strategy_declines_after_repeated_invalid_json(self):
        complete = SequenceComplete("", "not json", "[]")
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                complete_fn=complete,
                log_fn=logs.append,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )

            proposal = strat(ctx)

            self.assertIsNone(proposal)
            self.assertEqual(complete.calls, 3)
            self.assertEqual(len(logs), 3)

    def test_current_parent_selection_does_not_read_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )

            def fail_if_selector_is_used(_ctx, _method):
                raise AssertionError("current parent selection should not use selector")

            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                parent_selection="current",
                parent_selector=fail_if_selector_is_used,
                build_prompt=lambda _version, _failures: "prompt",
                complete_fn=fake_complete,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )
            proposal = strat(ctx)

            self.assertEqual(proposal.base, exp.current().hash)

    def test_non_current_parent_selection_uses_recipe_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            selected = exp.current().hash
            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                parent_selection="best",
                parent_selector=lambda _ctx, method: selected,
                build_prompt=lambda _version, _failures: "prompt",
                complete_fn=fake_complete,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )
            proposal = strat(ctx)
            self.assertEqual(proposal.base, selected)

    def test_parent_selector_must_return_version_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                parent_selection="best",
                parent_selector=lambda _ctx, method: f"selected-{method}",
                build_prompt=lambda _version, _failures: "prompt",
                complete_fn=fake_complete,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )
            with self.assertRaisesRegex(ValueError, "version hash"):
                strat(ctx)

    def test_non_current_parent_selection_requires_recipe_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            exp = Experiment(
                ws, rollout=lambda v, s: [], seed={"agent/agent_program.py": "X = 0\n"}
            )
            strat = model_program_strategy(
                provider=object(),
                prefix="agent/",
                parent_selection="best",
                build_prompt=lambda _version, _failures: "prompt",
                complete_fn=fake_complete,
            )
            ctx = Context(
                runs=(),
                current=exp.current(),
                workspace=ws,
                decisions=(),
                reward=lambda r: 0.0,
            )
            with self.assertRaisesRegex(ValueError, "parent_selector"):
                strat(ctx)


if __name__ == "__main__":
    unittest.main()
