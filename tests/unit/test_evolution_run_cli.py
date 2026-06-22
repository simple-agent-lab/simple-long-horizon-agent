from __future__ import annotations

import contextlib
import io
import json
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

    def _register_demo_factories(
        self,
        *,
        on_run: Callable[..., None] | None = None,
        strategy_factory: Callable[..., Any] | None = None,
        agent_program: str = "def build_agent(**kwargs): pass\n",
    ) -> None:
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
                default_files={"agent_program.py": agent_program},
                artifact_key=artifact_key,
            )
        )
        registry.BACKENDS["fake"] = lambda **_args: FakeBackend(on_run=on_run)
        registry.STORES["local_dir"] = lambda root, **_args: LocalDirStore(root)
        registry.STRATEGIES["model_program"] = strategy_factory or (
            lambda **_args: lambda _ctx: None
        )
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

    def test_missing_config_exits_nonzero(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main([])

        self.assertNotEqual(raised.exception.code, 0)

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
        self.assertIn(f"run root: {root.resolve() / 'override-demo'}", output)

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
            run_root = root.resolve() / "monitor-demo"
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

    def test_execute_writes_heldout_performance_summary(self) -> None:
        from simple_agent_lab.evals import RESULT_KEY
        from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
        from simple_agent_lab.evolution.types import Proposal

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heldout_path = root / "heldout.jsonl"
            heldout_path.write_text(
                '{"instance_id": "h1"}\n{"instance_id": "h2"}\n',
                encoding="utf-8",
            )
            config = self._write_demo_config(root, run_id="measured-demo")
            text = config.read_text(encoding="utf-8")
            text = text.replace(
                "instances:\n"
                "  train:\n"
                "    id: train\n"
                "    path: " + str(root / "train.jsonl") + "\n",
                "instances:\n"
                "  train:\n"
                "    id: train\n"
                "    path: " + str(root / "train.jsonl") + "\n"
                "  heldout:\n"
                "    id: heldout\n"
                "    path: " + str(heldout_path) + "\n",
            )
            text = text.replace("  api_kind: openai-chat\n", "  api_kind: fake\n")
            text = text.replace("  rounds: 2\n", "  rounds: 1\n")
            text = text.replace(
                "  baseline_heldout: false\n", "  baseline_heldout: true\n"
            )
            text = text.replace("  final_heldout: false\n", "  final_heldout: true\n")
            config.write_text(text, encoding="utf-8")

            def on_run(_spec: Any, bound: Any) -> None:
                package = json.loads(bound.get(AGENT_PACKAGE_KEY).decode("utf-8"))
                reward = 1.0 if "better" in package["agent_program.py"] else 0.0
                bound.put(
                    RESULT_KEY,
                    (
                        json.dumps({"reward": reward, "resolved": reward > 0.0}) + "\n"
                    ).encode("utf-8"),
                )

            proposed = False

            def strategy_factory(**_args: Any) -> Callable[..., Any]:
                def strategy(_ctx: Any) -> Proposal | None:
                    nonlocal proposed
                    if proposed:
                        return None
                    proposed = True
                    return Proposal(
                        {
                            "agent/agent_program.py": "def build_agent(**kwargs):\n"
                            "    return 'better'\n"
                        },
                        note="make the demo agent better",
                    )

                return strategy

            self._register_demo_factories(
                on_run=on_run,
                strategy_factory=strategy_factory,
                agent_program="def build_agent(**kwargs):\n    return 'baseline'\n",
            )

            output = self._run_cli(["--config", str(config), "--execute"])
            summary_path = root / "measured-demo" / "evaluation" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [row["label"] for row in summary["evaluations"]], ["baseline", "final"]
        )
        self.assertEqual(summary["evaluations"][0]["metrics"]["reward_mean"], 0.0)
        self.assertEqual(summary["evaluations"][0]["metrics"]["resolved"], 0)
        self.assertEqual(summary["evaluations"][1]["metrics"]["reward_mean"], 1.0)
        self.assertEqual(summary["evaluations"][1]["metrics"]["resolved"], 2)
        self.assertEqual(summary["delta"]["reward_mean"], 1.0)
        self.assertEqual(summary["delta"]["resolved"], 2)
        self.assertIn("heldout baseline: reward=0.000 resolved=0/2", output)
        self.assertIn("heldout final: reward=1.000 resolved=2/2", output)
        self.assertIn("heldout delta: reward=+1.000 resolved=+2", output)

    def test_execute_requires_heldout_when_evaluation_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._register_demo_factories()
            config = self._write_demo_config(root)
            text = config.read_text(encoding="utf-8")
            text = text.replace("  api_kind: openai-chat\n", "  api_kind: fake\n")
            text = text.replace(
                "  baseline_heldout: false\n", "  baseline_heldout: true\n"
            )
            config.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "heldout"):
                main(["--config", str(config), "--execute"])


if __name__ == "__main__":
    unittest.main()
