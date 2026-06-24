from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.types import Context

from recipes.ahe.analyzer import AnalysisResult
from recipes.ahe.surface import ahe_harness_surface
from recipes.ahe.strategy import ahe_model_strategy


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class RecordingAnalyzer:
    def __init__(self, overview: str = "# Overview\nanalysis.\n") -> None:
        self.overview = overview
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> AnalysisResult:
        self.calls.append((args, kwargs))
        output_dir = Path(args[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        overview_path = output_dir / "overview.md"
        overview_path.write_text(self.overview, encoding="utf-8")
        index_path = output_dir / "index.json"
        index_data = {
            "version": str(args[2].hash),
            "run_count": len(args[1]),
            "failed_count": len(args[1]),
            "patterns": [],
            "details": {},
        }
        index_path.write_text(
            json.dumps(index_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        detail_dir = output_dir / "detail"
        detail_dir.mkdir(parents=True, exist_ok=True)
        return AnalysisResult(
            overview_path=overview_path,
            detail_dir=detail_dir,
            index_path=index_path,
            overview=self.overview,
            index=index_data,
        )


def _make_context(workspace: Path) -> tuple[Experiment, Context]:
    surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)
    experiment = Experiment(
        workspace,
        rollout=lambda _version, _slice: [],
        seed=surface.seed_files(),
    )
    context = Context(
        runs=(),
        current=experiment.current(),
        workspace=workspace,
        decisions=(),
        reward=lambda _run: 0.0,
    )
    return experiment, context


class AheStrategyTest(unittest.TestCase):
    def test_ahe_strategy_runs_analyzer_writes_manifest_and_returns_proposal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, ctx = _make_context(workspace)
            current_hash = experiment.current().hash
            analyzer = RecordingAnalyzer()

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "tighten shell validation evidence",
                            "evidence": ["analysis: shallow validation pattern"],
                            "manifest": {
                                "round": 1,
                                "base_version": "abc",
                                "changes": [
                                    {
                                        "id": "chg-1",
                                        "type": "improvement",
                                        "component": "system_prompt",
                                        "files": ["harness/systemprompt.md"],
                                        "failure_pattern": "shallow validation",
                                        "root_cause": "agent trusted existence checks",
                                        "targeted_fix": (
                                            "require evaluator-like validation"
                                        ),
                                        "predicted_fixes": ["i1"],
                                        "risk_tasks": [],
                                        "why_this_component": "global behavior rule",
                                    }
                                ],
                            },
                            "edits": {
                                "harness/systemprompt.md": (
                                    "You are an AHE harness agent.\n"
                                    "Use bash for focused local work, keep changes small, and explain what you observed.\n"
                                    "\n"
                                    "Add strict validation before trusting filesystem checks.\n"
                                )
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY),
                editable_components=("system_prompt",),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

            manifest_path = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(
                (
                    root / "ahe" / "rounds" / "round_001" / "analysis" / "overview.md"
                ).is_file()
            )

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.kind, "ahe_harness")
        self.assertIn("harness/systemprompt.md", proposal.edits)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], current_hash)
        self.assertIn(
            "manifest:ahe/rounds/round_001/change_manifest.json",
            proposal.evidence,
        )
        self.assertTrue(analyzer.calls)

    def test_ahe_strategy_rejects_disallowed_and_unchanged_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, ctx = _make_context(workspace)
            current_hash = experiment.current().hash
            analyzer = RecordingAnalyzer()
            unchanged_prompt = experiment.current().read("harness/systemprompt.md")

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "copy seed",
                            "evidence": ["analysis: shallow validation pattern"],
                            "manifest": {
                                "round": 99,
                                "base_version": "abc",
                                "changes": [],
                            },
                            "edits": {
                                "harness/systemprompt.md": unchanged_prompt,
                                "harness/tools/bash.py": "x = 1\n",
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY),
                editable_components=("system_prompt",),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

            manifest_path = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(
                (
                    root / "ahe" / "rounds" / "round_001" / "analysis" / "overview.md"
                ).is_file()
            )

        self.assertIsNone(proposal)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], current_hash)

    def test_ahe_strategy_skips_missing_knowledge_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            _, ctx = _make_context(workspace)
            analyzer = RecordingAnalyzer()
            missing_knowledge = root / "missing.md"

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "keep going",
                            "evidence": ["analysis: shallow validation pattern"],
                            "manifest": {
                                "round": 1,
                                "base_version": "abc",
                                "changes": [],
                            },
                            "edits": {
                                "harness/systemprompt.md": (
                                    "You are an AHE harness agent.\n"
                                    "Use bash for focused local work, keep changes small, and explain what you observed.\n"
                                    "\n"
                                    "Add strict validation before trusting filesystem checks.\n"
                                )
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY),
                editable_components=("system_prompt",),
                knowledge_paths=(str(missing_knowledge),),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.kind, "ahe_harness")
        self.assertTrue(analyzer.calls)


if __name__ == "__main__":
    unittest.main()
