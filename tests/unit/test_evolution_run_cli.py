from __future__ import annotations

import contextlib
import io
import json
import os
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
  name: demo_source
  editable_components: [everything]
  artifact_key: input/source.json
  default: demo_source
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
        from simple_agent_lab.evolution.surface import AgentSurface, SurfaceComponent

        previous: dict[tuple[str, str], Callable[..., Any] | None] = {
            ("suite", "demo_suite"): registry.SUITES.get("demo_suite"),
            ("surface", "demo_source"): registry.SURFACES.get("demo_source"),
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
        registry.SURFACES["demo_source"] = lambda *, default, artifact_key, **_args: (
            AgentSurface(
                id="demo_source",
                name="Demo source",
                description="Demo source tree.",
                entrypoint="src/demo/agent_program.py:build_agent",
                default_files={"src/demo/agent_program.py": agent_program},
                artifact_key=artifact_key,
                components=(
                    SurfaceComponent(
                        id="everything",
                        name="Everything",
                        description="All demo source.",
                        paths=("src/demo/**",),
                        validators=("path_allowed", "python_source", "python_syntax"),
                    ),
                ),
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
        self.assertIn("surface: demo_source", output)
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

    def test_dotenv_is_loaded_before_provider_seed_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_model = os.environ.pop("OPENAI_MODEL", None)
            self.addCleanup(
                lambda: (
                    os.environ.__setitem__("OPENAI_MODEL", old_model)
                    if old_model is not None
                    else os.environ.pop("OPENAI_MODEL", None)
                )
            )
            self._register_demo_factories()
            config = self._write_demo_config(root)
            dotenv = root / "demo.env"
            dotenv.write_text("OPENAI_MODEL=dotenv-model\n", encoding="utf-8")
            text = config.read_text(encoding="utf-8")
            text = text.replace("  dotenv: .env\n", f"  dotenv: {dotenv}\n")
            config.write_text(text, encoding="utf-8")

            self._run_cli(["--config", str(config)])
            provider_path = next(
                (root / "demo" / "evolution" / "versions").glob("*/provider.json")
            )
            provider = json.loads(provider_path.read_text(encoding="utf-8"))

        self.assertEqual(provider["model"], "dotenv-model")

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
            text = text.replace("name: demo_source", "name: unregistered_surface")
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
                source = json.loads(bound.get("input/source.json").decode("utf-8"))
                reward = 1.0 if "better" in source["src/demo/agent_program.py"] else 0.0
                bound.put(
                    RESULT_KEY,
                    (
                        json.dumps(
                            {
                                "reward": reward,
                                "resolved": reward > 0.0,
                            }
                        )
                        + "\n"
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
                            "src/demo/agent_program.py": "def build_agent(**kwargs):\n"
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
        self.assertIn("[progress] run start id=measured-demo", output)
        self.assertIn("rounds=1", output)
        self.assertIn("train=1", output)
        self.assertIn("heldout=2", output)
        self.assertIn("[progress] round start index=1 total=1", output)
        self.assertIn("[progress] decision accepted", output)
        self.assertIn("baseline_reward=0.000", output)
        self.assertIn("candidate_reward=1.000", output)
        self.assertIn("delta=+1.000", output)
        self.assertIn("[progress] heldout complete label=baseline", output)
        self.assertIn("[progress] heldout complete label=final", output)
        self.assertIn("[progress] run complete id=measured-demo", output)
        self.assertIn("decisions=1", output)
        self.assertIn("accepted=1", output)

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
