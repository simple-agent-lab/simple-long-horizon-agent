"""JSON model config — the structured sibling of the single-provider env scheme.

Once a deployment runs more than a couple of models, the ``<ALIAS>_MODEL`` /
``<ALIAS>_AUTH_TOKEN`` / ``<ALIAS>_BASE_URL`` / ``<ALIAS>_API_KIND`` env sprawl
gets hard to read. This module loads the *same* alias → `Provider` mapping from
a small JSON document instead, so a `ModelRegistry` can be built from a file or
env var (see `simple_long_horizon_agent.llm.registry`). JSON, not YAML, keeps the
dependency set at zero — `Provider` is already described as JSON-serializable.

Schema (every field optional except a resolved ``model`` per alias)::

    {
      "defaults": {                      // shared base; each alias merges over it
        "base_url": "https://gateway/v1",
        "api_kind": "openai-chat",       // openai-chat | openai-responses | anthropic-messages
        "auth_token_env": "OPENAI_AUTH_TOKEN",
        "reasoning_effort": "medium"     // minimal | low | medium | high | xhigh
      },
      "aliases": {
        "strong": { "model": "big-model", "api_kind": "anthropic-messages" },
        "fast":   { "model": "small-model" }
      }
    }

Cloud sandboxes that cannot easily create a separate ``models.json`` file can
put the same JSON text in ``MODEL_CONFIG_JSON``; `ModelRegistry.load` checks it
before ``MODEL_CONFIG``.

Secrets: prefer ``auth_token_env`` — the file names the env var that *holds* the
token, so the file itself carries no secret and is safe to commit. An inline
``auth_token`` is allowed for quick local runs but is a footgun (gitignore your
file); loading one emits a `UserWarning` and stashes the value in a private
process env var so the adapter — which reads ``os.environ`` via
`Provider.api_key_env` — still finds it. Omit both for keyless local endpoints
(Ollama, a `fake` adapter).
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, cast

from .env import API_KIND_CHOICES
from .provider import (
    REASONING_EFFORTS,
    ApiKind,
    Provider,
    ReasoningEffort,
    api_kind_defaults,
)

# Env vars `ModelRegistry.load` reads to choose model-config mode without the
# call site branching. `MODEL_CONFIG_JSON` exists for cloud sandboxes where
# writing a separate models.json file is awkward; it uses the same schema.
MODEL_CONFIG_ENV = "MODEL_CONFIG"
MODEL_CONFIG_JSON_ENV = "MODEL_CONFIG_JSON"

# Per-alias keys the schema accepts. A typo is a hard error, not a silent
# ignore, so a misspelled "base_ur" doesn't quietly fall back to the default.
_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "api_kind",
        "base_url",
        "auth_token_env",
        "auth_token",
        "reasoning_effort",
        "temperature",
        "max_tokens",
        "context_window",
        "replay_reasoning",
    }
)
_TOP_KEYS: frozenset[str] = frozenset({"defaults", "aliases"})

# Prefix for the synthetic env var an inline `auth_token` is stashed under.
_INLINE_AUTH_PREFIX = "_SAL_MODELCONFIG_"


def _declared_keys(mapping: Mapping[str, Any]) -> set[str]:
    """The mapping's real keys — JSON has no comments, so a leading-``_`` key is
    a doc/comment the loader ignores (top-level and per-alias). Lets the example
    file explain itself without a separate README round-trip."""
    return {k for k in mapping if not k.startswith("_")}


def load_model_config_text(
    text: str,
    *,
    source_name: str = MODEL_CONFIG_JSON_ENV,
    missing_exc: Callable[[str], BaseException] = SystemExit,
) -> dict[str, dict[str, Any]]:
    """Read and validate JSON model config from a string.

    This is the env-var sibling of `load_model_config`: same schema, same
    strict validation, no filesystem dependency. `source_name` is only for clean
    error messages.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise missing_exc(f"Invalid JSON in {source_name}: {exc}") from exc
    return _load_model_config_raw(raw, path=Path(source_name), missing_exc=missing_exc)


def load_model_config(
    path: str | Path,
    *,
    missing_exc: Callable[[str], BaseException] = SystemExit,
) -> dict[str, dict[str, Any]]:
    """Read and validate the JSON file into ``{alias: merged_spec}``.

    Merges each alias spec over ``defaults`` and validates the shape — unknown
    top-level or per-alias keys, an unknown ``api_kind``/``reasoning_effort``, a
    non-object alias, or both ``auth_token`` and ``auth_token_env`` on one alias
    all raise ``missing_exc`` (``SystemExit`` by default, for a clean CLI message
    with no traceback). The returned specs are plain dicts, not `Provider`s, so
    `provider_from_spec` stays the single place that touches the environment.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise missing_exc(f"Model config file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise missing_exc(f"Invalid JSON in {config_path}: {exc}") from exc
    return _load_model_config_raw(raw, path=config_path, missing_exc=missing_exc)


def _load_model_config_raw(
    raw: Any,
    *,
    path: Path,
    missing_exc: Callable[[str], BaseException],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise missing_exc(f"Model config {path} must be a JSON object.")

    unknown_top = _declared_keys(raw) - _TOP_KEYS
    if unknown_top:
        raise missing_exc(
            f"Unknown top-level keys in {path}: {sorted(unknown_top)}; "
            f"expected {sorted(_TOP_KEYS)}."
        )

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise missing_exc(f"'defaults' in {path} must be an object.")
    aliases = raw.get("aliases", {})
    if not isinstance(aliases, dict) or not aliases:
        raise missing_exc(f"'aliases' in {path} must be a non-empty object.")

    _validate_spec_keys(defaults, where="defaults", path=path, exc=missing_exc)

    merged: dict[str, dict[str, Any]] = {}
    for alias, spec in aliases.items():
        if not isinstance(spec, dict):
            raise missing_exc(f"alias {alias!r} in {path} must be an object.")
        _validate_spec_keys(spec, where=alias, path=path, exc=missing_exc)
        combined = {**defaults, **spec}
        _validate_values(alias, combined, path=path, exc=missing_exc)
        merged[alias] = combined
    return merged


def _validate_spec_keys(
    spec: dict[str, Any],
    *,
    where: str,
    path: Path,
    exc: Callable[[str], BaseException],
) -> None:
    unknown = _declared_keys(spec) - _SPEC_KEYS
    if unknown:
        raise exc(
            f"Unknown keys in {where} of {path}: {sorted(unknown)}; "
            f"allowed {sorted(_SPEC_KEYS)}."
        )


def _validate_values(
    alias: str,
    spec: dict[str, Any],
    *,
    path: Path,
    exc: Callable[[str], BaseException],
) -> None:
    if not spec.get("model"):
        raise exc(f"alias {alias!r} in {path} is missing a 'model'.")
    api_kind = spec.get("api_kind", "openai-chat")
    if api_kind not in API_KIND_CHOICES:
        raise exc(
            f"alias {alias!r} in {path} has unsupported api_kind {api_kind!r}; "
            f"expected one of {sorted(API_KIND_CHOICES)}."
        )
    effort = spec.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise exc(
            f"alias {alias!r} in {path} has unsupported reasoning_effort "
            f"{effort!r}; expected one of {list(REASONING_EFFORTS)}."
        )
    if spec.get("auth_token") and spec.get("auth_token_env"):
        raise exc(
            f"alias {alias!r} in {path} sets both 'auth_token' and "
            f"'auth_token_env'; choose one."
        )


def provider_from_spec(
    alias: str,
    spec: dict[str, Any],
    *,
    env: MutableMapping[str, str] | None = None,
) -> Provider:
    """Build one `Provider` from a validated, defaults-merged spec.

    The only side effect is for an inline ``auth_token``: the literal is written
    to a private env var (``_SAL_MODELCONFIG_<ALIAS>_AUTH_TOKEN``) in `env`
    (``os.environ`` by default) and `Provider.api_key_env` is pointed at it, so
    the adapter's existing ``os.environ`` key lookup is unchanged. A
    ``auth_token_env`` is used as-is; absent both, the provider carries no key
    (keyless local / fake endpoints).
    """
    target_env = os.environ if env is None else env
    api_kind = cast(ApiKind, spec.get("api_kind", "openai-chat"))

    inline = spec.get("auth_token")
    if inline:
        warnings.warn(
            f"alias {alias!r} uses an inline 'auth_token'; prefer "
            f"'auth_token_env' and keep the secret out of the config file.",
            stacklevel=2,
        )
        auth_env = f"{_INLINE_AUTH_PREFIX}{alias.upper()}_AUTH_TOKEN"
        target_env[auth_env] = str(inline)
    else:
        auth_env = spec.get("auth_token_env") or ""

    effort = spec.get("reasoning_effort")
    reasoning = cast(ReasoningEffort, effort) if effort else None

    # Per-protocol defaults (Responses rejects temperature, caps output tokens)
    # live with the `Provider` definition; the file may override either.
    defaults = api_kind_defaults(api_kind)
    temperature = spec.get("temperature", defaults.default_temperature)
    max_tokens = spec.get("max_tokens", defaults.default_max_tokens)

    return Provider(
        id=alias,
        api=api_kind,
        model=str(spec["model"]),
        base_url=spec.get("base_url") or None,
        api_key_env=auth_env,
        default_temperature=temperature,
        default_max_tokens=max_tokens,
        default_reasoning=reasoning,
        context_window=spec.get("context_window"),
        replay_reasoning=spec.get("replay_reasoning", True),
    )
