from __future__ import annotations

import unittest

import simple_agent_lab.evolution as evolution
from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.registry import Use


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


if __name__ == "__main__":
    unittest.main()
