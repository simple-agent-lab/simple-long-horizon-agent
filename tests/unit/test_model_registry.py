"""Named model-alias registry over `provider_from_env`.

The registry maps aliases (``strong`` / ``fast`` / ...) to providers built from
``<ALIAS>_*`` env vars, each falling back to the base ``OPENAI_*`` set. See ADR
model-alias-registry.
"""

from __future__ import annotations

import unittest

from simple_agent_lab.llm import (
    DEFAULT_MODEL_ALIASES,
    FAST_ALIAS,
    STRONG_ALIAS,
    FAKE_PROVIDER,
    ModelRegistry,
    env_names_for,
)


class EnvNamesForTest(unittest.TestCase):
    def test_uppercases_and_builds_the_four_names(self) -> None:
        names = env_names_for("fast")
        self.assertEqual(names.model, "FAST_MODEL")
        self.assertEqual(names.auth, "FAST_AUTH_TOKEN")
        self.assertEqual(names.base_url, "FAST_BASE_URL")
        self.assertEqual(names.api_kind, "FAST_API_KIND")


class ModelRegistryFromEnvTest(unittest.TestCase):
    def test_base_openai_env_resolves_every_alias_to_one_model(self) -> None:
        # Minimal single-model setup: both aliases resolve to OPENAI_* via the
        # fallback, so existing one-model deployments need no new env.
        env = {"OPENAI_MODEL": "big", "OPENAI_AUTH_TOKEN": "tok"}
        registry = ModelRegistry.from_env(env=env)
        self.assertEqual(set(registry.aliases()), set(DEFAULT_MODEL_ALIASES))
        self.assertEqual(registry.get(STRONG_ALIAS).model, "big")
        self.assertEqual(registry.get(FAST_ALIAS).model, "big")
        # The fallback also supplies the auth env name the adapter will read.
        self.assertEqual(registry.get(FAST_ALIAS).api_key_env, "OPENAI_AUTH_TOKEN")

    def test_alias_specific_model_overrides_only_that_alias(self) -> None:
        env = {
            "OPENAI_MODEL": "big",
            "OPENAI_AUTH_TOKEN": "tok",
            "FAST_MODEL": "small",
        }
        registry = ModelRegistry.from_env(env=env)
        self.assertEqual(registry.get(STRONG_ALIAS).model, "big")
        self.assertEqual(registry.get(FAST_ALIAS).model, "small")
        # FAST_AUTH_TOKEN unset -> the gateway token is shared from OPENAI_*.
        self.assertEqual(registry.get(FAST_ALIAS).api_key_env, "OPENAI_AUTH_TOKEN")

    def test_alias_specific_auth_base_url_and_api_kind(self) -> None:
        env = {
            "OPENAI_MODEL": "big",
            "OPENAI_AUTH_TOKEN": "tok",
            "FAST_MODEL": "small",
            "FAST_AUTH_TOKEN": "fast-tok",
            "FAST_BASE_URL": "https://fast.invalid/v1",
            "FAST_API_KIND": "openai-responses",
        }
        fast = ModelRegistry.from_env(env=env).get(FAST_ALIAS)
        self.assertEqual(fast.model, "small")
        self.assertEqual(fast.api, "openai-responses")
        self.assertEqual(fast.api_key_env, "FAST_AUTH_TOKEN")
        self.assertEqual(fast.base_url, "https://fast.invalid/v1")

    def test_custom_aliases(self) -> None:
        env = {"OPENAI_MODEL": "m", "OPENAI_AUTH_TOKEN": "t", "CHEAP_MODEL": "c"}
        registry = ModelRegistry.from_env(("strong", "cheap"), env=env)
        self.assertEqual(registry.aliases(), ("strong", "cheap"))
        self.assertEqual(registry.get("cheap").model, "c")

    def test_shared_reasoning_effort_applies_to_each_alias(self) -> None:
        env = {
            "OPENAI_MODEL": "big",
            "OPENAI_AUTH_TOKEN": "tok",
            "REASONING_EFFORT": "high",
        }
        registry = ModelRegistry.from_env(env=env)
        self.assertEqual(registry.get(STRONG_ALIAS).default_reasoning, "high")
        self.assertEqual(registry.get(FAST_ALIAS).default_reasoning, "high")

    def test_no_base_and_no_alias_env_raises(self) -> None:
        # Nothing configured at all -> SystemExit naming the alias.
        with self.assertRaises(SystemExit):
            ModelRegistry.from_env(env={})

    def test_fallback_none_requires_explicit_alias_env(self) -> None:
        # With fallback disabled, OPENAI_* no longer satisfies an alias.
        env = {"OPENAI_MODEL": "big", "OPENAI_AUTH_TOKEN": "tok", "STRONG_MODEL": "s"}
        with self.assertRaises(SystemExit):
            # `fast` has no FAST_* and no fallback -> missing.
            ModelRegistry.from_env(env=env, fallback=None)
        # Only `strong` configured -> that single alias builds.
        registry = ModelRegistry.from_env(
            ("strong",),
            env={"STRONG_MODEL": "s", "STRONG_AUTH_TOKEN": "k"},
            fallback=None,
        )
        self.assertEqual(registry.get("strong").model, "s")


class ModelRegistryProgrammaticTest(unittest.TestCase):
    def test_get_unknown_alias_lists_configured_ones(self) -> None:
        registry = ModelRegistry({"strong": FAKE_PROVIDER})
        self.assertIs(registry.get("strong"), FAKE_PROVIDER)
        with self.assertRaises(KeyError) as ctx:
            registry.get("fast")
        self.assertIn("strong", str(ctx.exception))
        self.assertIn("fast", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
