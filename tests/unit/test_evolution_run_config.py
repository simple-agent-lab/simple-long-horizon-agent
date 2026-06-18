from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.run_config import load_self_evolving_config


CONFIG = """
run:
  id: demo
  output_root: evals/out/demo
  execute: false
  reset: false
  dotenv: .env
suite:
  name: swebench
  args:
    dataset_name: demo-dataset
surface:
  name: python_agent_package
  editable_components: [everything]
  artifact_key: input/agent_package.json
  default: simple_agent_package
instances:
  train:
    id: train
    path: train.jsonl
execution:
  backend:
    name: fake
  store:
    name: local_dir
  parallel: 1
  max_turns: 3
model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
strategy:
  name: model_program
  args:
    system_prompt: demo
evolution:
  algorithm: simple
  rounds: 2
  criterion:
    name: promote_not_worse
    args:
      dim: reward
evaluation:
  baseline_heldout: false
  final_heldout: false
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: false
"""


class RunConfigTest(unittest.TestCase):
    def _write_demo_config(self, root: Path, config_text: str = CONFIG) -> Path:
        train_path = root / "train.jsonl"
        train_path.write_text('{"instance_id": "i1"}\n', encoding="utf-8")
        path = root / "config.yaml"
        path.write_text(
            config_text.replace("output_root: evals/out/demo", f"output_root: {root}")
            .replace("swebench", "demo_suite")
            .replace("train.jsonl", str(train_path)),
            encoding="utf-8",
        )
        return path

    def _register_demo_factories(self, strategy_factory) -> None:
        from simple_agent_lab.evals import FakeBackend, LocalDirStore
        from simple_agent_lab.evals.protocols import LaunchSpec
        from simple_agent_lab.evolution import registry
        from simple_agent_lab.evolution.surface import python_agent_surface

        class DemoSuite:
            name = "demo"
            container_module = "demo.container"

            def launch_spec(self, instance):
                return LaunchSpec(image="python:3.11", workdir="/work")

            def task_input(self, instance):
                return dict(instance)

            def eval_inputs(self, instance):
                return None

        registry.SUITES["demo_suite"] = lambda **_args: DemoSuite()
        registry.SURFACES["python_agent_package"] = (
            lambda *, default, artifact_key, **_args: python_agent_surface(
                default_files={"agent_program.py": "def build_agent(**kwargs): pass\n"},
                artifact_key=artifact_key,
            )
        )
        registry.BACKENDS["fake"] = lambda **_args: FakeBackend(on_run=None)
        registry.STORES["local_dir"] = lambda root, **_args: LocalDirStore(root)
        registry.STRATEGIES["model_program"] = strategy_factory
        self.addCleanup(registry.SUITES.pop, "demo_suite", None)
        self.addCleanup(registry.SURFACES.pop, "python_agent_package", None)
        self.addCleanup(registry.BACKENDS.pop, "fake", None)
        self.addCleanup(registry.STORES.pop, "local_dir", None)
        self.addCleanup(registry.STRATEGIES.pop, "model_program", None)

    def test_load_self_evolving_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(CONFIG, encoding="utf-8")

            config = load_self_evolving_config(path)

        self.assertEqual(config.run.id, "demo")
        self.assertEqual(config.suite.name, "swebench")
        self.assertEqual(config.surface.editable_components, ("everything",))
        self.assertEqual(config.evolution.rounds, 2)

    def test_load_self_evolving_config_keeps_scalar_editable_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                CONFIG.replace(
                    "  editable_components: [everything]",
                    "  editable_components: prompts",
                ),
                encoding="utf-8",
            )

            config = load_self_evolving_config(path)

        self.assertEqual(config.surface.editable_components, ("prompts",))

    def test_missing_required_section_names_the_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("run: {id: demo}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "suite"):
                load_self_evolving_config(path)

    def test_build_self_evolving_run_with_registered_factories(self) -> None:
        from simple_agent_lab.evolution.run_config import build_self_evolving_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_demo_config(root)
            self._register_demo_factories(
                lambda **_args: lambda _ctx: None,
            )

            built = build_self_evolving_run(load_self_evolving_config(path))

            self.assertEqual(built.suite.name, "demo")
            self.assertEqual(built.train.id, "train")
            self.assertEqual(built.editable_components, ("everything",))
            self.assertEqual(
                built.experiment.workspace,
                root / "demo" / "evolution",
            )
            self.assertEqual(
                built.experiment.current().files(),
                ("agent/agent_program.py", "provider.json"),
            )

    def test_build_self_evolving_run_with_model_program_strategy(self) -> None:
        from simple_agent_lab.evolution.components.strategy import (
            model_program_strategy,
        )
        from simple_agent_lab.evolution.run_config import build_self_evolving_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_demo_config(root)
            self._register_demo_factories(model_program_strategy)

            built = build_self_evolving_run(load_self_evolving_config(path))

        self.assertTrue(callable(built.strategy))


if __name__ == "__main__":
    unittest.main()
