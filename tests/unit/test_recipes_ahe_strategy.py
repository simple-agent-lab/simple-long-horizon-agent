from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.types import Context, Run

from recipes.ahe.analyzer import AnalysisResult
from recipes.ahe.surface import ahe_harness_surface
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

    def test_ahe_strategy_scores_runs_before_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment, _ctx = _make_context(workspace)
            run_dir = root / "runs" / "baseline" / "i1"
            out_dir = run_dir / "out"
            out_dir.mkdir(parents=True)
            (out_dir / "result.json").write_text(
                json.dumps({"eval_log": "raw swe-bench log"}),
                encoding="utf-8",
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
                            "edits": {"harness/systemprompt.md": "scored\n"},
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

        self.assertIsNotNone(proposal)
        self.assertEqual(reward_calls, ["i1"])
        self.assertEqual(analyzer.calls[0][1]["run_scores"], {"i1": {"reward": 0.0}})

    def test_ahe_strategy_uses_next_unused_round_after_no_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            _, ctx = _make_context(workspace)
            analyzer = RecordingAnalyzer()
            calls = {"count": 0}

            def complete_fn(_request: object) -> FakeResponse:
                calls["count"] += 1
                if calls["count"] == 1:
                    return FakeResponse("not json")
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "second attempt",
                            "evidence": [],
                            "manifest": {"changes": []},
                            "edits": {"harness/systemprompt.md": "second\n"},
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

            first = strategy(ctx)
            second = strategy(ctx)

            first_manifest = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            second_manifest = (
                root / "ahe" / "rounds" / "round_002" / "change_manifest.json"
            )
            first_manifest_exists = first_manifest.exists()
            second_manifest_exists = second_manifest.is_file()
            second_data = json.loads(second_manifest.read_text(encoding="utf-8"))

        self.assertIsNone(first)
        self.assertFalse(first_manifest_exists)
        self.assertIsNotNone(second)
        self.assertTrue(second_manifest_exists)
        self.assertEqual(second_data["round"], 2)

    def test_ahe_strategy_canonicalizes_prompt_bounds_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            experiment = Experiment(
                workspace,
                rollout=lambda _version, _slice: [],
                seed={
                    "harness/zzz.md": "Z" * 2000,
                    "harness/aaa.md": "A" * 2000,
                    "harness/systemprompt.md": "S" * 2000,
                },
            )
            ctx = Context(
                runs=(),
                current=experiment.current(),
                workspace=workspace,
                decisions=tuple(
                    type(
                        "DecisionLike",
                        (),
                        {
                            "id": f"dec-{index}",
                            "outcome": "accepted",
                            "reason": "R" * 500,
                        },
                    )()
                    for index in range(7)
                ),
                reward=lambda _run: 0.0,
            )
            analysis_overview = "# Overview\n" + ("V" * 2000) + "\n"
            analysis_index = {"noise": "O" * 3000, "version": experiment.current().hash}
            knowledge_path = root / "knowledge.md"
            knowledge_path.write_text("K" * 2000, encoding="utf-8")
            captured_prompt: dict[str, str] = {}

            def analyzer_fn(*args: object, **kwargs: object) -> AnalysisResult:
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
                self.assertIn("### harness/aaa.md", prompt)
                self.assertIn("### harness/zzz.md", prompt)
                self.assertLess(
                    prompt.index("### harness/aaa.md"),
                    prompt.index("### harness/zzz.md"),
                )
                self.assertIn("K" * 200, prompt)
                self.assertNotIn("K" * (MAX_KNOWLEDGE_CHARS + 20), prompt)
                self.assertIn("A" * 200, prompt)
                self.assertNotIn("A" * (MAX_HARNESS_FILE_CHARS + 20), prompt)
                self.assertIn("V" * 200, prompt)
                self.assertNotIn("V" * (MAX_ANALYSIS_OVERVIEW_CHARS + 20), prompt)
                self.assertIn("O" * 200, prompt)
                self.assertNotIn("O" * (MAX_ANALYSIS_INDEX_CHARS + 20), prompt)
                self.assertIn("- ... 2 earlier decisions omitted", prompt)
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "prompt bounded",
                            "evidence": [],
                            "manifest": {"changes": []},
                            "edits": {"harness/systemprompt.md": "new prompt\n"},
                        }
                    )
                )

            strategy = ahe_model_strategy(
                provider=object(),
                surface=ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY),
                editable_components=("system_prompt",),
                knowledge_paths=(str(knowledge_path),),
                complete_fn=complete_fn,
                analyzer_fn=analyzer_fn,
            )

            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        self.assertEqual(captured_prompt["text"].count("### harness/"), 3)

    def test_ahe_strategy_normalizes_malformed_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            _, ctx = _make_context(workspace)
            analyzer = RecordingAnalyzer()

            def complete_fn(_request: object) -> FakeResponse:
                return FakeResponse(
                    json.dumps(
                        {
                            "note": "normalize manifest",
                            "evidence": ["ok"],
                            "manifest": {
                                "round": 999,
                                "base_version": "wrong",
                                "changes": [
                                    "drop-me",
                                    {
                                        "id": 7,
                                        "type": 1,
                                        "component": ["system_prompt"],
                                        "files": "harness/systemprompt.md",
                                        "failure_pattern": None,
                                        "root_cause": 42,
                                        "targeted_fix": {"x": 1},
                                        "predicted_fixes": ("i1", 2),
                                        "risk_tasks": None,
                                        "why_this_component": False,
                                    },
                                    {
                                        "component": "tool_implementations",
                                        "files": ["harness/tools/bash.py"],
                                        "predicted_fixes": ["i2"],
                                        "risk_tasks": ["r1", 2],
                                        "why_this_component": "global rule",
                                    },
                                ],
                            },
                            "edits": {"harness/systemprompt.md": "normalized\n"},
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

        self.assertIsNotNone(proposal)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], ctx.current.hash)
        self.assertEqual(len(manifest["changes"]), 2)
        self.assertEqual(manifest["changes"][0]["id"], "7")
        self.assertEqual(manifest["changes"][0]["type"], "1")
        self.assertEqual(manifest["changes"][0]["component"], "unknown")
        self.assertEqual(manifest["changes"][0]["files"], [])
        self.assertEqual(manifest["changes"][0]["failure_pattern"], "")
        self.assertEqual(manifest["changes"][0]["root_cause"], "42")
        self.assertEqual(manifest["changes"][0]["targeted_fix"], "")
        self.assertEqual(manifest["changes"][0]["predicted_fixes"], ["i1"])
        self.assertEqual(manifest["changes"][0]["risk_tasks"], [])
        self.assertEqual(manifest["changes"][0]["why_this_component"], "False")
        self.assertEqual(manifest["changes"][1]["id"], "chg-3")
        self.assertEqual(manifest["changes"][1]["component"], "tool_implementations")
        self.assertEqual(manifest["changes"][1]["predicted_fixes"], ["i2"])
        self.assertEqual(manifest["changes"][1]["risk_tasks"], ["r1"])
        self.assertEqual(manifest["changes"][1]["why_this_component"], "global rule")


if __name__ == "__main__":
    unittest.main()
