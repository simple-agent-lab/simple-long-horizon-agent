from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from recipes.ahe.strategy import ahe_agent_strategy
from recipes.ahe.surface import ahe_harness_surface
from recipes.dgm.evolve import dgm_agentic_strategy
from recipes.dgm.swebench import AGENT_PREFIX, seed_files
from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evolution import Experiment
from simple_agent_lab.evolution.types import Context
from recipes.ahe.analyzer import AnalysisResult


class AgenticRecipeStrategyTest(unittest.TestCase):
    def test_dgm_strategy_runs_parent_agent_package_as_meta_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "evolution"
            experiment = Experiment(
                workspace,
                rollout=lambda _version, _slice: [],
                seed={
                    **seed_files(model="fake-model", api_kind="fake"),
                    AGENT_PREFIX + "agent_program.py": _dgm_meta_agent_program(),
                },
            )
            ctx = Context(
                runs=(),
                current=experiment.current(),
                workspace=workspace,
                decisions=(),
            )

            strategy = dgm_agentic_strategy(provider=object(), max_turns=3)
            proposal = strategy(ctx)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.base, ctx.current.hash)
        self.assertEqual(proposal.kind, "dgm_agentic")
        self.assertIn(AGENT_PREFIX + "prompts.py", proposal.edits)
        self.assertIn("DGM_CONTEXT", proposal.edits[AGENT_PREFIX + "prompts.py"])
        self.assertIn("dgm-parent-agent-ran", proposal.evidence)

    def test_ahe_strategy_runs_evolve_agent_over_harness_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "evolution"
            surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)
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
                editable_components=("system_prompt",),
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
        self.assertEqual(proposal.kind, "ahe_harness")
        self.assertEqual(proposal.base, ctx.current.hash)
        self.assertIn("harness/systemprompt.md", proposal.edits)
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
            (self.cwd / "harness" / "systemprompt.md").write_text(
                "You are an AHE harness agent.\nUse evaluator-like validation.\n",
                encoding="utf-8",
            )
            (self.cwd / "change_manifest.json").write_text(
                json.dumps(
                    {
                        "changes": [
                            {
                                "id": "chg-1",
                                "type": "improvement",
                                "component": "system_prompt",
                                "files": ["harness/systemprompt.md"],
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


def _dgm_meta_agent_program() -> str:
    return """\
from pathlib import Path


class ParentMetaAgent:
    def __init__(self, cwd: Path):
        self.cwd = cwd

    def run(self, task: str, *, max_turns: int):
        def events():
            context = self.cwd / "SELF_IMPROVEMENT_CONTEXT.md"
            if not context.is_file():
                return
            (self.cwd / "agent" / "prompts.py").write_text(
                "DGM_CONTEXT = " + repr(context.read_text(encoding="utf-8")[:80]) + "\\n",
                encoding="utf-8",
            )
            yield object()
        return object(), events()


def build_agent(*, provider, cwd, base_system_prompt):
    return ParentMetaAgent(Path(cwd))
"""


if __name__ == "__main__":
    unittest.main()
