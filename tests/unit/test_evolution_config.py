from __future__ import annotations

import unittest

from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.config import Use


class RegistryTest(unittest.TestCase):
    def test_builtin_criterion_resolves_by_name(self) -> None:
        crit = registry.build("criterion", Use("improve", dim="reward"))
        base = {"i1": {"reward": 0.0}}
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
