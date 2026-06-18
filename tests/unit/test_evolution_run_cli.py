from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.run import main


CONFIG = """
run:
  id: demo
  output_root: {output_root}
  execute: false
  reset: false
  dotenv: .env
suite:
  name: demo_suite
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
    path: {train_path}
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


class EvolutionRunCliTest(unittest.TestCase):
    def _write_demo_config(self, root: Path, *, run_id: str = "demo") -> Path:
        train_path = root / "train.jsonl"
        train_path.write_text('{"instance_id": "i1"}\n', encoding="utf-8")
        path = root / "config.yaml"
        path.write_text(
            CONFIG.format(output_root=root, train_path=train_path).replace(
                "id: demo", f"id: {run_id}", 1
            ),
            encoding="utf-8",
        )
        return path

    def _register_demo_factories(self) -> None:
        from simple_agent_lab.evals import FakeBackend, LocalDirStore
        from simple_agent_lab.evals.protocols import LaunchSpec
        from simple_agent_lab.evolution import registry
        from simple_agent_lab.evolution.surface import python_agent_surface

        previous: dict[tuple[str, str], Callable[..., Any] | None] = {
            ("suite", "demo_suite"): registry.SUITES.get("demo_suite"),
            ("surface", "python_agent_package"): registry.SURFACES.get(
                "python_agent_package"
            ),
            ("backend", "fake"): registry.BACKENDS.get("fake"),
            ("store", "local_dir"): registry.STORES.get("local_dir"),
            ("strategy", "model_program"): registry.STRATEGIES.get("model_program"),
        }

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
        registry.STRATEGIES["model_program"] = lambda **_args: lambda _ctx: None
        self.addCleanup(self._restore_registry, previous)

    def _restore_registry(
        self, previous: dict[tuple[str, str], Callable[..., Any] | None]
    ) -> None:
        from simple_agent_lab.evolution import registry

        tables = {
            "suite": registry.SUITES,
            "surface": registry.SURFACES,
            "backend": registry.BACKENDS,
            "store": registry.STORES,
            "strategy": registry.STRATEGIES,
        }
        for (category, name), factory in previous.items():
            table = tables[category]
            if factory is None:
                table.pop(name, None)
            else:
                table[name] = factory

    def _run_cli(self, argv: list[str]) -> str:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(argv)
        self.assertEqual(code, 0)
        return stream.getvalue()

    def test_dry_run_prints_useful_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._register_demo_factories()
            config = self._write_demo_config(root)

            output = self._run_cli(["--config", str(config)])

        self.assertIn("dry-run self-evolving plan", output)
        self.assertIn("run id: demo", output)
        self.assertIn("suite: demo_suite", output)
        self.assertIn("surface: python_agent_package", output)
        self.assertIn("editable components: everything", output)
        self.assertIn("train: train", output)
        self.assertIn("train count: 1", output)
        self.assertIn("rounds: 2", output)

    def test_run_id_override_appears_in_dry_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._register_demo_factories()
            config = self._write_demo_config(root)

            output = self._run_cli(
                ["--config", str(config), "--run-id", "override-demo"]
            )

        self.assertIn("run id: override-demo", output)
        self.assertIn(f"run root: {root / 'override-demo'}", output)

    def test_reset_clears_stale_state_before_build_and_keeps_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._register_demo_factories()
            config = self._write_demo_config(root, run_id="reset-demo")
            run_root = root / "reset-demo"
            stale = run_root / "stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("old state\n", encoding="utf-8")

            self._run_cli(["--config", str(config), "--reset"])

            self.assertFalse(stale.exists())
            self.assertTrue((run_root / "evolution").is_dir())
            self.assertTrue(
                (run_root / "evolution" / "pointers" / "current.json").is_file()
            )

    def test_monitor_with_run_id_override_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_demo_config(root)
            text = config.read_text(encoding="utf-8")
            text = text.replace("name: demo_suite", "name: unregistered_suite")
            text = text.replace(
                "name: python_agent_package", "name: unregistered_surface"
            )
            text = text.replace("name: fake", "name: unregistered_backend")
            text = text.replace("name: local_dir", "name: unregistered_store")
            text = text.replace("name: model_program", "name: unregistered_strategy")
            config.write_text(text, encoding="utf-8")
            run_root = root / "monitor-demo"
            stale = run_root / "stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("old state\n", encoding="utf-8")

            output = self._run_cli(
                [
                    "--config",
                    str(config),
                    "--run-id",
                    "monitor-demo",
                    "--monitor",
                    "--reset",
                ]
            )

            self.assertEqual(f"monitor: {run_root}\n", output)
            self.assertTrue(stale.is_file())
            self.assertFalse((run_root / "evolution").exists())


if __name__ == "__main__":
    unittest.main()
