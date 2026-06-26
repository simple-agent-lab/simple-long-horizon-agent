from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals import RESULT_KEY, FakeBackend, LaunchSpec, LocalDirStore
from simple_agent_lab.evals.protocols import RunSpec
from simple_agent_lab.evolution.components.rollout import rollout_from_suite
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.surface import AgentSurface, SurfaceComponent
from simple_agent_lab.evolution.types import Manifest, Slice


class DemoSuite:
    name = "demo"
    container_module = "demo.container"

    def launch_spec(self, instance):
        return LaunchSpec(image="python:3.11", workdir="/work")

    def task_input(self, instance):
        return dict(instance)

    def eval_inputs(self, instance):
        return None


class RolloutFromSuiteTest(unittest.TestCase):
    def test_rollout_stages_surface_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "evolution"
            runs_root = root / "runs"
            surface = AgentSurface(
                id="demo_source",
                name="Demo source",
                description="Demo source tree.",
                entrypoint="src/demo/agent_program.py:build_agent",
                default_files={
                    "src/demo/agent_program.py": "def build_agent(**kwargs): pass\n"
                },
                artifact_key="input/source.json",
                components=(
                    SurfaceComponent(
                        id="everything",
                        name="Everything",
                        description="All demo source.",
                        paths=("src/demo/**",),
                    ),
                ),
            )
            version = store.stage(
                workspace,
                base=None,
                edits=surface.seed_files(),
                manifest=Manifest(producer="test"),
            )

            def on_run(spec: RunSpec, bound) -> None:
                bound.put(
                    RESULT_KEY,
                    (json.dumps({"reward": 1.0}) + "\n").encode("utf-8"),
                )

            rollout = rollout_from_suite(
                suite=DemoSuite(),
                surface=surface,
                backend=FakeBackend(on_run=on_run),
                store=LocalDirStore(runs_root),
                runs_root=runs_root,
            )

            runs = rollout(version, Slice("train", ({"instance_id": "i1"},)))

            self.assertEqual(len(runs), 1)
            staged = runs[0].dir / "input/source.json"
            payload = json.loads(staged.read_text(encoding="utf-8"))
            self.assertIn("src/demo/agent_program.py", payload)
            self.assertTrue(runs[0].ok)


if __name__ == "__main__":
    unittest.main()
