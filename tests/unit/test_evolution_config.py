from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import RESULT_KEY, TRACE_KEY, FakeBackend, LocalDirStore
from simple_agent_lab.evals.protocols import LaunchSpec, RunSpec
from simple_agent_lab.evolution.config import (
    CriterionConfig,
    EvolutionRunConfig,
    ExecutionConfig,
    EvaluationConfig,
    InstanceFileConfig,
    InstancesConfig,
    ModelConfig,
    NamedConfig,
    RunConfig,
    SelfEvolvingConfig,
    StrategyConfig,
    SurfaceConfig,
    build_self_evolving_run,
)
from simple_agent_lab.evolution.source_tree import CANDIDATE_SOURCE_CONTAINER_SRC
import simple_agent_lab.evolution as evolution
from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.registry import Use


class _ConfigSuite:
    name = "config-suite"
    container_module = "config.container"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="demo:latest", workdir="/work")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


class RegistryTest(unittest.TestCase):
    def test_use_belongs_to_registry(self) -> None:
        self.assertEqual(Use.__module__, "simple_agent_lab.evolution.registry")
        self.assertNotIn("Config", evolution.__all__)
        self.assertFalse(hasattr(evolution, "Config"))

    def test_builtin_criterion_resolves_by_name(self) -> None:
        crit = registry.build("criterion", Use("improve", dim="reward"))
        base = {"i1": {"reward": 0.0}}
        cand = {"i1": {"reward": 1.0}}
        self.assertTrue(crit(base, cand).accepted)

    def test_promote_not_worse_resolves_by_name(self) -> None:
        crit = registry.build("criterion", Use("promote_not_worse", dim="reward"))
        base = {"i1": {"reward": 1.0}}
        cand = {"i1": {"reward": 1.0}}
        self.assertTrue(crit(base, cand).accepted)

    def test_register_and_build_custom(self) -> None:
        registry.REWARDS["myreward"] = lambda: lambda run: 1.0
        fn = registry.build("reward", Use("myreward"))
        self.assertEqual(fn(object()), 1.0)

    def test_unknown_name_lists_options(self) -> None:
        with self.assertRaises(KeyError) as cm:
            registry.build("criterion", Use("nope"))
        self.assertIn("improve", str(cm.exception))


class SourceTreeConfigBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo_root = self.root / "repo"
        package = self.repo_root / "src" / "simple_agent_lab"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 'seed'\n", encoding="utf-8")
        self.instances = self.root / "train.jsonl"
        self.instances.write_text('{"instance_id": "i1"}\n', encoding="utf-8")
        self._old_model = os.environ.get("OPENAI_MODEL")
        os.environ["OPENAI_MODEL"] = "fake-model"
        self.addCleanup(self._restore_model_env)
        self._snapshots = {
            "suite": registry.SUITES.get("config_suite"),
            "surface": registry.SURFACES.get("source_tree"),
            "backend": registry.BACKENDS.get("fake_backend"),
            "store": registry.STORES.get("local_dir"),
            "strategy": registry.STRATEGIES.get("source_tree_agent"),
        }
        self.addCleanup(self._restore_registries)

    def _restore_model_env(self) -> None:
        if self._old_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = self._old_model

    def _restore_registries(self) -> None:
        targets = (
            (registry.SUITES, "config_suite", self._snapshots["suite"]),
            (registry.SURFACES, "source_tree", self._snapshots["surface"]),
            (registry.BACKENDS, "fake_backend", self._snapshots["backend"]),
            (registry.STORES, "local_dir", self._snapshots["store"]),
            (registry.STRATEGIES, "source_tree_agent", self._snapshots["strategy"]),
        )
        for table, name, original in targets:
            if original is None:
                table.pop(name, None)
            else:
                table[name] = original

    def test_source_tree_config_stages_candidate_source_and_pythonpath(self) -> None:
        from simple_agent_lab.evolution.components.repo_strategy import (
            source_tree_agent_strategy,
        )
        from simple_agent_lab.evolution.source_tree import source_tree_agent_surface

        seen_pythonpath: dict[str, tuple[str, ...]] = {}
        staged: dict[str, bytes] = {}

        def on_run(spec: RunSpec, bound) -> None:
            seen_pythonpath[spec.instance_id] = spec.pythonpath
            staged[spec.instance_id] = bound.get(
                "input/source_tree/src/simple_agent_lab/__init__.py"
            )
            bound.put(TRACE_KEY, b'{"events": []}\n')
            bound.put(RESULT_KEY, json.dumps({"reward": 1.0}).encode("utf-8"))

        registry.SUITES["config_suite"] = lambda **_args: _ConfigSuite()
        registry.SURFACES["source_tree"] = lambda **args: source_tree_agent_surface(
            repo_root=self.repo_root,
            **args,
        )
        registry.BACKENDS["fake_backend"] = lambda **_args: FakeBackend(on_run=on_run)
        registry.STORES["local_dir"] = lambda root, **_args: LocalDirStore(root)
        registry.STRATEGIES["source_tree_agent"] = source_tree_agent_strategy

        built = build_self_evolving_run(
            SelfEvolvingConfig(
                run=RunConfig(
                    id="config-source-tree",
                    output_root=str(self.root / "out"),
                ),
                suite=NamedConfig("config_suite"),
                surface=SurfaceConfig(
                    name="source_tree",
                    editable_components=("everything",),
                    default="current_source_tree",
                    artifact_key="source_tree",
                ),
                instances=InstancesConfig(
                    train=InstanceFileConfig("train", str(self.instances))
                ),
                execution=ExecutionConfig(
                    backend=NamedConfig("fake_backend"),
                    store=NamedConfig("local_dir"),
                ),
                model=ModelConfig(
                    api_kind="openai-chat",
                    model_env="OPENAI_MODEL",
                    api_key_env="OPENAI_AUTH_TOKEN",
                ),
                strategy=StrategyConfig("source_tree_agent"),
                evolution=EvolutionRunConfig(
                    algorithm="simple",
                    rounds=1,
                    criterion=CriterionConfig("promote_not_worse", {"dim": "reward"}),
                ),
                evaluation=EvaluationConfig(),
            )
        )

        built.rollout(built.experiment.current(), built.train)

        self.assertEqual(
            seen_pythonpath["i1"],
            (CANDIDATE_SOURCE_CONTAINER_SRC,),
        )
        self.assertEqual(staged["i1"], b"VALUE = 'seed'\n")


if __name__ == "__main__":
    unittest.main()
