"""Model registry: aliases resolved from a JSON file, or one OPENAI_* provider.

Multi-model comes from a JSON config file (`llm.config`); single-model falls back
to the base ``OPENAI_*`` provider that ``main`` uses. See ADRs
model-alias-registry and model-config-file.
"""

from __future__ import annotations

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_agent_lab.llm import (
    DEFAULT_MODEL_ALIASES,
    DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS,
    FAKE_PROVIDER,
    MODEL_CONFIG_JSON_ENV,
    ModelRegistry,
)


class ModelRegistryProgrammaticTest(unittest.TestCase):
    def test_get_unknown_alias_lists_configured_ones(self) -> None:
        registry = ModelRegistry({"strong": FAKE_PROVIDER})
        self.assertIs(registry.get("strong"), FAKE_PROVIDER)
        with self.assertRaises(KeyError) as ctx:
            registry.get("fast")
        self.assertIn("strong", str(ctx.exception))
        self.assertIn("fast", str(ctx.exception))


class ModelRegistryFromFileTest(unittest.TestCase):
    def _write(self, config: dict) -> Path:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "models.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_aliases_merge_over_defaults(self) -> None:
        path = self._write(
            {
                "defaults": {
                    "base_url": "https://gw/v1",
                    "auth_token_env": "OPENAI_AUTH_TOKEN",
                    "reasoning_effort": "high",
                },
                "aliases": {
                    "strong": {"model": "big", "api_kind": "anthropic-messages"},
                    "fast": {"model": "small"},
                },
            }
        )
        registry = ModelRegistry.from_file(path)
        self.assertEqual(set(registry.aliases()), {"strong", "fast"})
        strong = registry.get("strong")
        self.assertEqual(strong.model, "big")
        self.assertEqual(strong.api, "anthropic-messages")
        # Inherited from defaults.
        self.assertEqual(strong.base_url, "https://gw/v1")
        self.assertEqual(strong.api_key_env, "OPENAI_AUTH_TOKEN")
        self.assertEqual(strong.default_reasoning, "high")
        # `fast` inherits api_kind/base_url/auth from defaults.
        fast = registry.get("fast")
        self.assertEqual(fast.model, "small")
        self.assertEqual(fast.api, "openai-chat")
        self.assertEqual(fast.base_url, "https://gw/v1")

    def test_responses_kind_defaults_temperature_and_max_tokens(self) -> None:
        path = self._write(
            {"aliases": {"r": {"model": "m", "api_kind": "openai-responses"}}}
        )
        provider = ModelRegistry.from_file(path).get("r")
        # The Responses API rejects temperature and caps output tokens.
        self.assertIsNone(provider.default_temperature)
        self.assertEqual(
            provider.default_max_tokens, DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS
        )

    def test_inline_token_warns_and_is_resolvable(self) -> None:
        path = self._write(
            {"aliases": {"strong": {"model": "m", "auth_token": "sk-secret"}}}
        )
        env: dict[str, str] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider = ModelRegistry.from_file(path, env=env).get("strong")
        self.assertTrue(any("inline" in str(w.message) for w in caught))
        # The literal is stashed under the synthetic env name the provider reads.
        self.assertEqual(env.get(provider.api_key_env), "sk-secret")
        self.assertNotEqual(provider.api_key_env, "")

    def test_keyless_alias_has_empty_api_key_env(self) -> None:
        path = self._write({"aliases": {"local": {"model": "llama3"}}})
        provider = ModelRegistry.from_file(path).get("local")
        self.assertEqual(provider.api_key_env, "")

    def test_underscore_keys_are_ignored_as_comments(self) -> None:
        path = self._write(
            {
                "_comment": "doc",
                "aliases": {"strong": {"_note": "x", "model": "big"}},
            }
        )
        self.assertEqual(ModelRegistry.from_file(path).get("strong").model, "big")

    def test_missing_model_raises(self) -> None:
        path = self._write({"aliases": {"strong": {"api_kind": "openai-chat"}}})
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file(path)

    def test_unknown_key_raises(self) -> None:
        path = self._write({"aliases": {"strong": {"model": "m", "base_ur": "typo"}}})
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file(path)

    def test_unknown_api_kind_raises(self) -> None:
        path = self._write({"aliases": {"strong": {"model": "m", "api_kind": "nope"}}})
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file(path)

    def test_both_auth_forms_raises(self) -> None:
        path = self._write(
            {
                "aliases": {
                    "strong": {
                        "model": "m",
                        "auth_token": "sk",
                        "auth_token_env": "X",
                    }
                }
            }
        )
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file("/no/such/models.json")

    def test_empty_aliases_raises(self) -> None:
        path = self._write({"defaults": {"base_url": "x"}, "aliases": {}})
        with self.assertRaises(SystemExit):
            ModelRegistry.from_file(path)

    def test_from_json_text_uses_file_schema(self) -> None:
        registry = ModelRegistry.from_json_text(
            json.dumps(
                {
                    "defaults": {"auth_token_env": "OPENAI_AUTH_TOKEN"},
                    "aliases": {"strong": {"model": "big"}},
                }
            )
        )
        provider = registry.get("strong")
        self.assertEqual(provider.model, "big")
        self.assertEqual(provider.api_key_env, "OPENAI_AUTH_TOKEN")

    def test_from_json_text_rejects_invalid_json(self) -> None:
        with self.assertRaises(SystemExit):
            ModelRegistry.from_json_text("{not json")


class ModelRegistryLoadTest(unittest.TestCase):
    def test_model_config_json_env_selects_inline_json(self) -> None:
        env = {
            MODEL_CONFIG_JSON_ENV: json.dumps(
                {"aliases": {"strong": {"model": "from-env-json"}}}
            ),
            "OPENAI_MODEL": "fallback",
            "OPENAI_AUTH_TOKEN": "tok",
        }
        registry = ModelRegistry.load(env=env)
        self.assertEqual(registry.aliases(), ("strong",))
        self.assertEqual(registry.get("strong").model, "from-env-json")

    def test_model_config_json_env_takes_priority_over_file(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "models.json"
        path.write_text(
            json.dumps({"aliases": {"strong": {"model": "from-file"}}}),
            encoding="utf-8",
        )
        env = {
            MODEL_CONFIG_JSON_ENV: json.dumps(
                {"aliases": {"strong": {"model": "from-env-json"}}}
            ),
            "MODEL_CONFIG": str(path),
        }
        registry = ModelRegistry.load(env=env)
        self.assertEqual(registry.get("strong").model, "from-env-json")

    def test_model_config_json_inline_token_stashes_in_passed_env(self) -> None:
        env = {
            MODEL_CONFIG_JSON_ENV: json.dumps(
                {"aliases": {"strong": {"model": "from-env-json", "auth_token": "sk"}}}
            )
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            provider = ModelRegistry.load(env=env).get("strong")
        self.assertEqual(env[provider.api_key_env], "sk")

    def test_model_config_env_selects_the_file(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "models.json"
        path.write_text(
            json.dumps({"aliases": {"strong": {"model": "from-file"}}}),
            encoding="utf-8",
        )
        registry = ModelRegistry.load(env={"MODEL_CONFIG": str(path)})
        self.assertEqual(registry.aliases(), ("strong",))
        self.assertEqual(registry.get("strong").model, "from-file")

    def test_no_model_config_maps_every_alias_to_the_openai_provider(self) -> None:
        # Single-model, main-compatible: every default alias resolves to the one
        # OPENAI_* provider, so code asking for strong/fast works with no file.
        registry = ModelRegistry.load(
            env={"OPENAI_MODEL": "big", "OPENAI_AUTH_TOKEN": "tok"}
        )
        self.assertEqual(set(registry.aliases()), set(DEFAULT_MODEL_ALIASES))
        for alias in DEFAULT_MODEL_ALIASES:
            self.assertEqual(registry.get(alias).model, "big")
            self.assertEqual(registry.get(alias).api_key_env, "OPENAI_AUTH_TOKEN")

    def test_no_config_and_no_openai_env_raises(self) -> None:
        with self.assertRaises(SystemExit):
            ModelRegistry.load(env={})
