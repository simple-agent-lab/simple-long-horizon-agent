"""Tests for the env-config registry (ADR centralized-env-config)."""

from __future__ import annotations

import unittest

from simple_agent_lab import config
from simple_agent_lab.config import EnvVar, as_bool, as_int


class EnvVarResolverTest(unittest.TestCase):
    def test_unset_or_blank_returns_default(self) -> None:
        var = EnvVar("X_NUM", 7, "agent.test", "doc", as_int())
        self.assertEqual(var.get({}), 7)
        self.assertEqual(var.get({"X_NUM": ""}), 7)
        self.assertEqual(var.get({"X_NUM": "   "}), 7)

    def test_env_value_is_parsed_and_stripped(self) -> None:
        var = EnvVar("X_NUM", 7, "agent.test", "doc", as_int())
        self.assertEqual(var.get({"X_NUM": " 12 "}), 12)

    def test_unparseable_value_falls_back_to_default(self) -> None:
        var = EnvVar("X_NUM", 7, "agent.test", "doc", as_int())
        self.assertEqual(var.get({"X_NUM": "not-an-int"}), 7)

    def test_explicit_default_overrides_declared_default(self) -> None:
        var = EnvVar("X_NUM", None, "agent.test", "doc", as_int())
        self.assertEqual(var.get({}, default=40), 40)
        # An env value still wins over the explicit default.
        self.assertEqual(var.get({"X_NUM": "5"}, default=40), 5)

    def test_as_int_clamps_to_minimum(self) -> None:
        var = EnvVar("X_NUM", 1, "agent.test", "doc", as_int(minimum=1))
        self.assertEqual(var.get({"X_NUM": "0"}), 1)
        self.assertEqual(var.get({"X_NUM": "-3"}), 1)

    def test_string_var_defaults_to_identity_parse(self) -> None:
        var = EnvVar("X_STR", "python", "eval.test", "doc")
        self.assertEqual(var.get({}), "python")
        self.assertEqual(var.get({"X_STR": " go "}), "go")

    def test_as_bool_truthy_set(self) -> None:
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("", False),
        ]:
            self.assertEqual(as_bool(raw), expected, raw)


class RegistryTest(unittest.TestCase):
    def test_registry_lists_declared_vars_with_grouped_hierarchy(self) -> None:
        names = {var.name for var in config.REGISTRY}
        self.assertIn("SAL_WORKFLOW_PDR_ROUNDS", names)
        self.assertIn("SWE_REPO_LANGUAGE", names)
        # Domains follow the dotted `domain.subsystem` hierarchy.
        domains = {var.group.split(".", 1)[0] for var in config.REGISTRY}
        self.assertEqual(domains, {"agent", "eval"})
        # Classification is by domain, not name prefix: SWE_PDR_* is agent.workflow.
        self.assertEqual(config.PDR_ROUNDS.group, "agent.workflow")
        self.assertEqual(config.REPO_LANGUAGE.group, "eval.swebench")

    def test_workflow_defaults_match_prior_inline_values(self) -> None:
        self.assertEqual(config.WORKER_MAX_TURNS.get({}), 40)
        self.assertEqual(config.LOOP_MAX_TURNS.get({}), 6)
        self.assertEqual(config.PDR_ROUNDS.get({}), 2)
        self.assertEqual(config.PDR_WIDTH.get({}), 3)


if __name__ == "__main__":
    unittest.main()
