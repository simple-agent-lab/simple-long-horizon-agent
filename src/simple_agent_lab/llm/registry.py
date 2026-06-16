"""Named model roles — a lightweight alias → `Provider` map.

Lets a deployment configure a few model *roles* by alias (``strong``, ``fast``,
...) and look them up, instead of every call site hardcoding one provider. A
strong model for the main agent plus a cheap ``fast`` one for compaction or
sub-agents is the motivating case.

Two configuration surfaces, both resolving to the same `ModelRegistry`:

- **Multi-model**: a JSON config file (see `simple_agent_lab.llm.config`). The
  file's ``aliases`` *are* the roles; ``ModelRegistry.from_file(path)``. Cloud
  sandboxes can pass the same JSON text in ``MODEL_CONFIG_JSON`` instead of
  creating a file.
- **Single-model**: the base ``OPENAI_*`` provider (``OPENAI_MODEL`` +
  ``OPENAI_AUTH_TOKEN`` etc., see `simple_agent_lab.llm.env`) — the contract
  ``main`` already uses. ``ModelRegistry.load()`` returns the JSON registry when
  ``MODEL_CONFIG_JSON`` or ``MODEL_CONFIG`` is set, else maps every default alias
  onto that one ``OPENAI_*`` provider, so code that asks for ``strong``/``fast``
  keeps working on a single-model deployment with no extra env::

      registry = ModelRegistry.load()
      main = make_llm_agent(name="main", provider=registry.get("strong"))
      summarizer = make_llm_agent(name="sum", provider=registry.get("fast"))

For full programmatic / per-alias control, build the providers yourself and pass
them in: ``ModelRegistry({"strong": p1, "fast": p2})``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import (
    MODEL_CONFIG_ENV,
    MODEL_CONFIG_JSON_ENV,
    load_model_config,
    load_model_config_text,
    provider_from_spec,
)
from .env import provider_from_env
from .provider import Provider

# The two roles the project ships by default. Any aliases work; these are just
# the well-known names `load` maps a single-model deployment onto.
STRONG_ALIAS = "strong"
FAST_ALIAS = "fast"
DEFAULT_MODEL_ALIASES: tuple[str, ...] = (STRONG_ALIAS, FAST_ALIAS)


@dataclass(frozen=True)
class ModelRegistry:
    """An immutable alias → `Provider` map with a strict `get`.

    Build it from JSON with `from_file` / `from_json_text`, via `load` (JSON or
    the base ``OPENAI_*`` provider), or pass a ready dict for programmatic /
    per-alias control. Lookups are strict: an unknown alias is a `KeyError`
    naming the configured ones, so a typo fails clearly instead of silently
    using the wrong model.
    """

    providers: Mapping[str, Provider]

    def get(self, alias: str) -> Provider:
        try:
            return self.providers[alias]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise KeyError(
                f"unknown model alias {alias!r}; configured aliases: {known}"
            ) from None

    def aliases(self) -> tuple[str, ...]:
        return tuple(self.providers)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        env: MutableMapping[str, str] | None = None,
        missing_exc: Callable[[str], BaseException] = SystemExit,
    ) -> "ModelRegistry":
        """Build the registry from a JSON model-config file (see `llm.config`).

        The file's ``aliases`` become the registry's aliases, each merged over the
        file's ``defaults``. Eager: every alias resolves now, so a bad spec fails
        at load, not at first use. `env` is where an inline ``auth_token`` is
        stashed (``os.environ`` by default); pass a dict in tests to keep the real
        environment clean.
        """
        specs = load_model_config(path, missing_exc=missing_exc)
        providers = {
            alias: provider_from_spec(alias, spec, env=env)
            for alias, spec in specs.items()
        }
        return cls(providers)

    @classmethod
    def from_json_text(
        cls,
        text: str,
        *,
        env: MutableMapping[str, str] | None = None,
        missing_exc: Callable[[str], BaseException] = SystemExit,
    ) -> "ModelRegistry":
        """Build the registry from JSON text using the same schema as a file."""
        specs = load_model_config_text(text, missing_exc=missing_exc)
        providers = {
            alias: provider_from_spec(alias, spec, env=env)
            for alias, spec in specs.items()
        }
        return cls(providers)

    @classmethod
    def load(
        cls,
        aliases: tuple[str, ...] = DEFAULT_MODEL_ALIASES,
        *,
        env: Mapping[str, str] | None = None,
        missing_exc: Callable[[str], BaseException] = SystemExit,
    ) -> "ModelRegistry":
        """One entry point: env JSON, then file JSON, then single ``OPENAI_*``.

        Lets a call site write ``ModelRegistry.load()`` and let the deployment
        decide *how* it is configured: set ``MODEL_CONFIG_JSON`` for cloud
        sandboxes, set ``MODEL_CONFIG=/path/models.json`` for file-based
        multi-model, or leave both unset to run the one ``main``-compatible
        ``OPENAI_*`` model under every role. Eager either way.
        """
        source = env if env is not None else os.environ
        target_env = (
            cast("MutableMapping[str, str]", env)
            if isinstance(env, MutableMapping)
            else None
        )
        config_json = source.get(MODEL_CONFIG_JSON_ENV, "")
        if config_json.strip():
            return cls.from_json_text(
                config_json,
                env=target_env,
                missing_exc=missing_exc,
            )
        config_path = source.get(MODEL_CONFIG_ENV, "")
        if config_path.strip():
            return cls.from_file(
                config_path.strip(), env=target_env, missing_exc=missing_exc
            )
        provider = provider_from_env(
            env=env,
            read_reasoning=True,
            label="the model provider",
            missing_exc=missing_exc,
        )
        return cls({alias: provider for alias in aliases})
