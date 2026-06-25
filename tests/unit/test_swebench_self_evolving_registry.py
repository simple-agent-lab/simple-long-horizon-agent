from __future__ import annotations

import unittest
from unittest.mock import patch

from simple_agent_lab.evolution import registry


class SimpleRecipeRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshots = {
            "swebench": registry.SUITES.get("swebench"),
            "source_tree": registry.SURFACES.get("source_tree"),
            "local_docker": registry.BACKENDS.get("local_docker"),
            "local_dir": registry.STORES.get("local_dir"),
            "source_tree_agent": registry.STRATEGIES.get("source_tree_agent"),
            "result_key": registry.REWARDS.get("result_key"),
        }
        self.addCleanup(self._restore_registries)

    def _restore_registries(self) -> None:
        targets = (
            (registry.SUITES, "swebench"),
            (registry.SURFACES, "source_tree"),
            (registry.BACKENDS, "local_docker"),
            (registry.STORES, "local_dir"),
            (registry.STRATEGIES, "source_tree_agent"),
            (registry.REWARDS, "result_key"),
        )
        for table, name in targets:
            original = self._snapshots[name]
            if original is None:
                table.pop(name, None)
            else:
                table[name] = original

    def test_registers_simple_recipe_factories(self) -> None:
        from recipes.simple.evolve import register_recipe_factories

        register_recipe_factories()

        self.assertIn("swebench", registry.SUITES)
        self.assertIn("source_tree", registry.SURFACES)
        self.assertNotIn("python_agent_package", registry.SURFACES)
        self.assertIn("local_docker", registry.BACKENDS)
        self.assertIn("local_dir", registry.STORES)
        self.assertIn("source_tree_agent", registry.STRATEGIES)
        self.assertNotIn("model_program", registry.STRATEGIES)
        self.assertEqual(
            registry.SUITES["swebench"]().container_module,
            "simple_agent_lab.evals.suites.swebench.evolving",
        )

    def test_registration_preserves_user_overrides(self) -> None:
        from recipes.simple.evolve import register_recipe_factories

        sentinel = object()
        registry.SUITES["swebench"] = sentinel

        register_recipe_factories()

        self.assertIs(registry.SUITES["swebench"], sentinel)

    def test_config_registration_uses_swebench_reward(self) -> None:
        from recipes.simple import evolve

        reward = object()
        with patch.object(
            evolve, "_swebench_reward_from_config", return_value=reward
        ) as build:
            evolve.register_recipe_factories("config.yaml")

        build.assert_called_once_with("config.yaml")
        self.assertIs(registry.REWARDS["result_key"](), reward)


if __name__ == "__main__":
    unittest.main()
