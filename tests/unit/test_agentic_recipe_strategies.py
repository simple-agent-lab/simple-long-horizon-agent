from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from recipes.ahe.strategy import ahe_agent_strategy
from recipes.dgm.evolve import dgm_agentic_strategy
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.source_tree import (
    SOURCE_ROOT,
    source_tree_agent_surface,
)
from simple_agent_lab.evolution.types import Context
from recipes.ahe.analyzer import AnalysisResult


class AgenticRecipeStrategyTest(unittest.TestCase):
    def test_dgm_strategy_runs_meta_agent_over_selected_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            package = repo_root / SOURCE_ROOT
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n")
            (package / "core.py").write_text("def run() -> str:\n    return 'ok'\n")
            workspace = root / "evolution"
            surface = source_tree_agent_surface(repo_root)
            experiment = Experiment(
                workspace,
                rollout=lambda _version, _slice: [],
                seed=surface.seed_files(),
            )
            ctx = Context(
                runs=(),
                current=experiment.current(),
                workspace=workspace,
                decisions=(),
            )

            fake_agent = SourceTreeFakeAgent()

            def agent_builder(**kwargs: object) -> SourceTreeFakeAgent:
                fake_agent.cwd = Path(kwargs["cwd"])
                return fake_agent

            strategy = dgm_agentic_strategy(
                provider=object(),
                repo_root=repo_root,
                surface=surface,
                editable_components=("everything",),
                agent_builder=agent_builder,
                max_turns=3,
            )
            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.base, ctx.current.hash)
        self.assertEqual(proposal.kind, "dgm_agentic")
        self.assertIn(SOURCE_ROOT + "/core.py", proposal.edits)
        self.assertIn("DGM_CONTEXT", proposal.edits[SOURCE_ROOT + "/core.py"])
        self.assertIn("dgm-source-tree-agent-ran", proposal.evidence)

    def test_ahe_strategy_runs_evolve_agent_over_source_tree_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            package = repo_root / SOURCE_ROOT
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n")
            (package / "core.py").write_text("def run() -> str:\n    return 'ok'\n")
            workspace = root / "evolution"
            surface = source_tree_agent_surface(repo_root)
            experiment = Experiment(
                workspace,
                rollout=lambda _version, _slice: [],
                seed=surface.seed_files(),
            )
            ctx = Context(
                runs=(),
                current=experiment.current(),
                workspace=workspace,
                decisions=(),
            )
            fake_agent = AheFakeAgent()

            def agent_builder(**kwargs: object) -> AheFakeAgent:
                fake_agent.cwd = Path(kwargs["cwd"])
                return fake_agent

            strategy = ahe_agent_strategy(
                provider=object(),
                surface=surface,
                editable_components=("agent_runtime",),
                agent_builder=agent_builder,
                analyzer_fn=deterministic_analyzer,
                max_turns=4,
            )

            proposal = strategy(ctx)

            manifest_path = (
                root / "ahe" / "rounds" / "round_001" / "change_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(fake_agent.events_consumed)
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.kind, "ahe_source_tree")
        self.assertEqual(proposal.base, ctx.current.hash)
        self.assertIn(SOURCE_ROOT + "/core.py", proposal.edits)
        self.assertEqual(manifest["round"], 1)
        self.assertEqual(manifest["base_version"], ctx.current.hash)
        self.assertEqual(manifest["changes"][0]["id"], "chg-1")
        self.assertIn(
            "manifest:ahe/rounds/round_001/change_manifest.json",
            proposal.evidence,
        )


class AheFakeAgent:
    def __init__(self) -> None:
        self.cwd: Path | None = None
        self.events_consumed = False

    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]:
        self.task = task
        self.max_turns = max_turns

        def events() -> Iterator[object]:
            assert self.cwd is not None
            overview = (
                self.cwd
                / "runs"
                / "iteration_001"
                / "input"
                / "analysis"
                / "overview.md"
            )
            if not overview.is_file():
                raise AssertionError("AHE evolve agent did not receive analysis")
            (self.cwd / SOURCE_ROOT / "core.py").write_text(
                "DGM_CONTEXT = 'analysis-guided source change'\n",
                encoding="utf-8",
            )
            (self.cwd / "change_manifest.json").write_text(
                json.dumps(
                    {
                        "changes": [
                            {
                                "id": "chg-1",
                                "type": "improvement",
                                "component": "agent_runtime",
                                "files": [SOURCE_ROOT + "/core.py"],
                                "failure_pattern": "shallow validation",
                                "root_cause": "agent trusted weak evidence",
                                "targeted_fix": "require evaluator-like validation",
                                "predicted_fixes": ["i1"],
                                "risk_tasks": [],
                                "why_this_component": "global workflow rule",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.events_consumed = True
            yield object()

        return object(), events()


def deterministic_analyzer(*args: object, **_kwargs: object) -> AnalysisResult:
    output_dir = Path(args[4])
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "overview.md"
    overview_path.write_text(
        "# Overview\nNeed stronger validation.\n", encoding="utf-8"
    )
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps({"patterns": [], "details": {}, "failed_count": 0}) + "\n",
        encoding="utf-8",
    )
    detail_dir = output_dir / "detail"
    detail_dir.mkdir(exist_ok=True)
    return AnalysisResult(
        overview_path=overview_path,
        detail_dir=detail_dir,
        index_path=index_path,
        overview=overview_path.read_text(encoding="utf-8"),
        index={"patterns": [], "details": {}, "failed_count": 0},
    )


class SourceTreeFakeAgent:
    def __init__(self) -> None:
        self.cwd: Path | None = None

    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]:
        def events() -> Iterator[object]:
            assert self.cwd is not None
            context = self.cwd / "SELF_EVOLUTION_CONTEXT.md"
            if not context.is_file():
                raise AssertionError("DGM source-tree agent did not receive context")
            (self.cwd / SOURCE_ROOT / "core.py").write_text(
                "DGM_CONTEXT = "
                + repr(context.read_text(encoding="utf-8")[:80])
                + "\n",
                encoding="utf-8",
            )
            yield object()

        return object(), events()


if __name__ == "__main__":
    unittest.main()
