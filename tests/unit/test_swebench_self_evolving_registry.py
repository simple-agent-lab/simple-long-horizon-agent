from __future__ import annotations

import unittest

from simple_agent_lab.evolution import registry


class SwebenchSelfEvolvingRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshots = {
            "swebench": registry.SUITES.get("swebench"),
            "python_agent_package": registry.SURFACES.get("python_agent_package"),
            "local_docker": registry.BACKENDS.get("local_docker"),
            "local_dir": registry.STORES.get("local_dir"),
            "model_program": registry.STRATEGIES.get("model_program"),
        }
        self.addCleanup(self._restore_registries)

    def _restore_registries(self) -> None:
        targets = (
            (registry.SUITES, "swebench"),
            (registry.SURFACES, "python_agent_package"),
            (registry.BACKENDS, "local_docker"),
            (registry.STORES, "local_dir"),
            (registry.STRATEGIES, "model_program"),
        )
        for table, name in targets:
            original = self._snapshots[name]
            if original is None:
                table.pop(name, None)
            else:
                table[name] = original

    def test_registers_swebench_self_evolving_factories(self) -> None:
        from evals.swebench.self_evolving import (
            register_swebench_self_evolving_factories,
        )

        register_swebench_self_evolving_factories()

        self.assertIn("swebench", registry.SUITES)
        self.assertIn("python_agent_package", registry.SURFACES)
        self.assertIn("local_docker", registry.BACKENDS)
        self.assertIn("local_dir", registry.STORES)
        self.assertIn("model_program", registry.STRATEGIES)

    def test_registration_preserves_user_overrides(self) -> None:
        from evals.swebench.self_evolving import (
            register_swebench_self_evolving_factories,
        )

        sentinel = object()
        registry.SUITES["swebench"] = sentinel

        register_swebench_self_evolving_factories()

        self.assertIs(registry.SUITES["swebench"], sentinel)


if __name__ == "__main__":
    unittest.main()
