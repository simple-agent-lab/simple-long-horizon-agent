from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.source_tree import (
    SOURCE_ROOT,
    source_tree_agent_surface,
)
from simple_agent_lab.evolution.types import Context, Run

from recipes.ahe.analyzer import AnalysisResult
from recipes.ahe.strategy import (
    MAX_ANALYSIS_INDEX_CHARS,
    MAX_ANALYSIS_OVERVIEW_CHARS,
    MAX_HARNESS_FILE_CHARS,
    MAX_KNOWLEDGE_CHARS,
    ahe_model_strategy,
)


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


def _make_context(workspace: Path) -> tuple[Experiment, Context, object]:
    repo_root = workspace.parent / "repo"
    package = repo_root / SOURCE_ROOT
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "core.py").write_text(
        "def run() -> str:\n    return 'ok'\n", encoding="utf-8"
    )
    (package / "tools").mkdir()
    (package / "tools" / "bash.py").write_text(
        "def bash_tool() -> str:\n    return 'bash'\n", encoding="utf-8"
    )
    surface = source_tree_agent_surface(repo_root)
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
    return experiment, context, surface


class AheStrategyTest(unittest.TestCase):
    def test_ahe_strategy_writes_manifest_and_returns_source_tree_proposal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, ctx, surface = _make_context(workspace)
            current_hash = experiment.current().hash
            analyzer = RecordingAnalyzer()

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "tighten runtime validation evidence",
                            "evidence": ["analysis: shallow validation pattern"],
                            "manifest": {
                                "round": 1,
                                "base_version": "abc",
                                "changes": [
                                    {
                                        "id": "chg-1",
                                        "type": "improvement",
                                        "component": "agent_runtime",
                                        "files": [SOURCE_ROOT + "/core.py"],
                                        "failure_pattern": "shallow validation",
                                        "root_cause": "agent trusted existence checks",
                                        "targeted_fix": "require evaluator-like validation",
                                        "predicted_fixes": ["i1"],
                                        "risk_tasks": [],
                                        "why_this_component": "global behavior rule",
                                    }
                                ],
                            },
                            "edits": {
                                SOURCE_ROOT
                                + "/core.py": "def run() -> str:\n    return 'better'\n"
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=surface,
                editable_components=("agent_runtime",),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

            manifest_path = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.kind, "ahe_source_tree")
        self.assertIn(SOURCE_ROOT + "/core.py", proposal.edits)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], current_hash)
        self.assertIn(
            "manifest:ahe/rounds/round_001/change_manifest.json",
            proposal.evidence,
        )
        self.assertTrue(analyzer.calls)

    def test_ahe_strategy_rejects_disallowed_and_unchanged_source_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, ctx, surface = _make_context(workspace)
            current_hash = experiment.current().hash
            analyzer = RecordingAnalyzer()
            unchanged_core = experiment.current().read(SOURCE_ROOT + "/core.py")

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "copy seed",
                            "evidence": [],
                            "manifest": {"round": 99, "base_version": "abc"},
                            "edits": {
                                SOURCE_ROOT + "/core.py": unchanged_core,
                                SOURCE_ROOT + "/tools/bash.py": "x = 1\n",
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=surface,
                editable_components=("agent_runtime",),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

            manifest_path = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIsNone(proposal)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], current_hash)

    def test_ahe_strategy_scores_runs_before_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, _ctx, surface = _make_context(workspace)
            run_dir = root / "runs" / "baseline" / "i1"
            out_dir = run_dir / "out"
            out_dir.mkdir(parents=True)
            (out_dir / "result.json").write_text(
                json.dumps({"eval_log": "raw swe-bench log"}), encoding="utf-8"
            )
            analyzer = RecordingAnalyzer()
            reward_calls = []
            ctx = Context(
                runs=(Run(run_dir),),
                current=experiment.current(),
                workspace=workspace,
                decisions=(),
                reward=lambda run: reward_calls.append(run.instance_id) or 0.0,
            )

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "score first",
                            "evidence": [],
                            "manifest": {"changes": []},
                            "edits": {
                                SOURCE_ROOT
                                + "/core.py": "def run() -> str:\n    return 'scored'\n"
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=surface,
                editable_components=("agent_runtime",),
                complete_fn=complete_fn,
                analyzer_fn=analyzer,
            )

            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        self.assertEqual(reward_calls, ["i1"])
        self.assertEqual(analyzer.calls[0][1]["run_scores"], {"i1": {"reward": 0.0}})

    def test_ahe_strategy_prompt_is_bounded_and_uses_surface_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            _experiment, ctx, surface = _make_context(workspace)
            analysis_overview = "# Overview\n" + ("V" * 2000) + "\n"
            analysis_index = {"noise": "O" * 3000, "version": ctx.current.hash}
            knowledge_path = root / "knowledge.md"
            knowledge_path.write_text("K" * 2000, encoding="utf-8")
            captured_prompt: dict[str, str] = {}

            def analyzer_fn(*args: object, **_kwargs: object) -> AnalysisResult:
                output_dir = Path(args[4])
                output_dir.mkdir(parents=True, exist_ok=True)
                overview_path = output_dir / "overview.md"
                overview_path.write_text(analysis_overview, encoding="utf-8")
                index_path = output_dir / "index.json"
                index_path.write_text(
                    json.dumps(analysis_index, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                detail_dir = output_dir / "detail"
                detail_dir.mkdir(parents=True, exist_ok=True)
                return AnalysisResult(
                    overview_path=overview_path,
                    detail_dir=detail_dir,
                    index_path=index_path,
                    overview=analysis_overview,
                    index=analysis_index,
                )

            def complete_fn(request: object) -> FakeResponse:
                prompt = request.messages[0].content[0].text
                captured_prompt["text"] = prompt
                self.assertIn("### " + SOURCE_ROOT + "/core.py", prompt)
                self.assertIn("K" * 200, prompt)
                self.assertNotIn("K" * (MAX_KNOWLEDGE_CHARS + 20), prompt)
                self.assertIn("V" * 200, prompt)
                self.assertNotIn("V" * (MAX_ANALYSIS_OVERVIEW_CHARS + 20), prompt)
                self.assertIn("O" * 200, prompt)
                self.assertNotIn("O" * (MAX_ANALYSIS_INDEX_CHARS + 20), prompt)
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "prompt bounded",
                            "evidence": [],
                            "manifest": {"changes": []},
                            "edits": {
                                SOURCE_ROOT
                                + "/core.py": "def run() -> str:\n    return 'new'\n"
                            },
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=surface,
                editable_components=("agent_runtime",),
                knowledge_paths=(str(knowledge_path),),
                complete_fn=complete_fn,
                analyzer_fn=analyzer_fn,
            )

            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        self.assertLessEqual(
            len(captured_prompt["text"].split("Current editable files:\n", 1)[1]),
            MAX_HARNESS_FILE_CHARS + 1200,
        )


if __name__ == "__main__":
    unittest.main()
