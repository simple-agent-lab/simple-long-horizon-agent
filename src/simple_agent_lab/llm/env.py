"""Provider configuration from the environment — the single source of truth.

Everything about *how a `Provider` is built from environment variables* lives
here: the canonical env-var names, the `.env` loader, the reasoning/effort and
request-header readers, and `provider_from_env` itself. Scripts, the TUI
gateway, the eval harnesses, and the live e2e tests all import from this module
instead of each re-declaring `OPENAI_MODEL_ENV = "OPENAI_MODEL"` and a bespoke
`build_*_provider()` (they used to — five near-identical copies that drifted).

The env contract, by convention of this project's gateway setup:

- ``OPENAI_MODEL`` / ``OPENAI_AUTH_TOKEN`` / ``OPENAI_BASE_URL`` — the model id,
  the bearer token (NOT ``OPENAI_API_KEY``; this project points at a gateway
  that authenticates with ``OPENAI_AUTH_TOKEN``), and an optional endpoint
  override.
- ``API_KIND`` — which wire adapter to use (``openai-chat`` default,
  ``openai-responses``, or ``anthropic-messages``). NOTE: the adapter and the
  endpoint must agree — pointing ``OPENAI_BASE_URL`` at an Anthropic-protocol
  URL while ``API_KIND`` stays ``openai-chat`` is the classic misconfig (it
  surfaces as a confusing 5xx from the gateway, not a clear error).
- ``OPENAI_SESSION_ID`` / ``OPENAI_LOG_ID`` — optional gateway trace headers.
- ``REASONING_EFFORT`` (provider-agnostic; legacy ``OPENAI_REASONING_EFFORT``).
- ``JUDGE_*`` — a parallel set for an eval grader, falling back to ``OPENAI_*``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .provider import (
    REASONING_EFFORTS,
    ApiKind,
    Provider,
    ReasoningEffort,
    api_kind_defaults,
)
from ..model_metadata import default_context_window_book

# --------------------------------------------------------------------------- #
# Canonical env-var names (declared once; everyone else imports these)
# --------------------------------------------------------------------------- #
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
# Provider-agnostic reasoning depth; the legacy OpenAI-specific name is honored.
REASONING_EFFORT_ENV = "REASONING_EFFORT"
OPENAI_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
# Eval-grader provider; each value falls back to the matching OPENAI_* one.
JUDGE_MODEL_ENV = "JUDGE_MODEL"
JUDGE_AUTH_ENV = "JUDGE_AUTH_TOKEN"
JUDGE_BASE_URL_ENV = "JUDGE_BASE_URL"
JUDGE_API_KIND_ENV = "JUDGE_API_KIND"

# Adapters that `provider_from_env` will build. A superset of what any single
# caller used before; validation only guards against typos.
API_KIND_CHOICES: tuple[str, ...] = (
    "openai-chat",
    "openai-responses",
    "anthropic-messages",
)
OPENAI_API_KIND_CHOICES = ("openai-chat", "openai-responses")

# The one deterministic, key-free provider — replaces a dozen copies of the
# `Provider(id="fake", api="fake", model="fake-model")` literal.
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def container_provider_env(
    provider: str,
    passthrough_names: Iterable[str],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Collect the shared OpenAI container env contract used by eval harnesses."""

    if provider != "openai":
        return {}
    source = os.environ if env is None else env
    names = (*passthrough_names, "NO_PROXY", "no_proxy")
    resolved = {name: source[name] for name in names if source.get(name)}
    missing = [
        name for name in (OPENAI_MODEL_ENV, OPENAI_AUTH_ENV) if name not in resolved
    ]
    if missing:
        raise SystemExit(
            "Missing required env vars for --provider openai: " + ", ".join(missing)
        )
    return resolved


def resolve_openai_api_kind(
    value: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the narrow OpenAI-protocol adapter choice used by eval harnesses."""

    source = os.environ if env is None else env
    api_kind = (value or source.get(API_KIND_ENV) or "openai-chat").strip()
    if api_kind not in OPENAI_API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported API_KIND {api_kind!r}; expected one of: "
            + ", ".join(OPENAI_API_KIND_CHOICES)
        )
    return api_kind


@dataclass(frozen=True)
class ProviderEnvNames:
    """The four env-var names one provider reads.

    Bundled so `provider_from_env` can take a primary set plus an optional
    fallback (the judge reads `JUDGE_*` and falls back to `OPENAI_*`) without
    string-prefix guessing — the `API_KIND` irregularity (`API_KIND` for the
    default provider vs `JUDGE_API_KIND` for the judge) is just data here.
    """

    model: str
    auth: str
    base_url: str
    api_kind: str


OPENAI_ENV = ProviderEnvNames(
    model=OPENAI_MODEL_ENV,
    auth=OPENAI_AUTH_ENV,
    base_url=OPENAI_BASE_URL_ENV,
    api_kind=API_KIND_ENV,
)
JUDGE_ENV = ProviderEnvNames(
    model=JUDGE_MODEL_ENV,
    auth=JUDGE_AUTH_ENV,
    base_url=JUDGE_BASE_URL_ENV,
    api_kind=JUDGE_API_KIND_ENV,
)


def load_dotenv(path: str | Path, *, environ: dict[str, str] | None = None) -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file without overriding the env.

    Tolerates ``export KEY=value``, ``#`` comments, blank lines, and quoted
    values. Existing environment values win (a `.env` only fills the gaps), so
    an explicit export on the command line always overrides the file.
    """
    target = os.environ if environ is None else environ
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key and key not in target:
            target[key] = value.strip().strip("'\"")


def reasoning_from_env(source: Mapping[str, str]) -> ReasoningEffort | None:
    """Read the normalized reasoning effort; the adapter maps it per-model.

    Honors the provider-agnostic ``REASONING_EFFORT`` and the legacy
    ``OPENAI_REASONING_EFFORT``. An unrecognized value is a hard error so a typo
    fails fast rather than reaching the model unmapped.
    """
    effort = (
        source.get(REASONING_EFFORT_ENV, "")
        or source.get(OPENAI_REASONING_EFFORT_ENV, "")
    ).strip()
    if not effort:
        return None
    if effort not in REASONING_EFFORTS:
        raise SystemExit(
            f"Unsupported reasoning effort {effort!r}; "
            f"expected one of {REASONING_EFFORTS}."
        )
    return cast(ReasoningEffort, effort)


def request_extra_from_env(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the gateway trace headers (session id / log id) request-extra."""
    source = env if env is not None else os.environ
    session_id = source.get(OPENAI_SESSION_ID_ENV, "").strip()
    log_id = source.get(OPENAI_LOG_ID_ENV, "").strip()
    if not session_id and not log_id:
        return {}
    return {
        "extra_headers": {
            "extra": json.dumps({"session_id": session_id}, separators=(",", ":")),
            "X-TT-logid": log_id,
        }
    }


def _pick(
    source: Mapping[str, str], primary: str, fallback: str | None
) -> tuple[str, str]:
    """Return ``(value, env_name_that_supplied_it)``, trying primary then fallback.

    The returned name defaults to ``primary`` even when empty, so a missing-env
    message and `Provider.api_key_env` name the variable the caller should set.
    """
    value = (source.get(primary) or "").strip()
    if value:
        return value, primary
    if fallback:
        alt = (source.get(fallback) or "").strip()
        if alt:
            return alt, fallback
    return "", primary


def provider_from_env(
    names: ProviderEnvNames = OPENAI_ENV,
    *,
    fallback: ProviderEnvNames | None = None,
    env: Mapping[str, str] | None = None,
    api_kind: str | None = None,
    default_temperature: float | None = 1.0,
    default_reasoning: ReasoningEffort | None = None,
    read_reasoning: bool = False,
    label: str = "the model provider",
    missing_exc: Callable[[str], BaseException] = SystemExit,
    reexport_auth: bool = False,
) -> Provider:
    """Build an OpenAI-compatible (or Anthropic) `Provider` from env vars.

    One function behind every former ``build_*_provider`` / ``*_provider_from_env``
    copy. Reads model/auth/base-url/api-kind from `names` (falling back to
    `fallback`, e.g. the judge → OPENAI_*). Knobs cover the few real differences
    between the old copies:

    - `api_kind`: explicit override; otherwise read from the api-kind env var,
      defaulting to ``"openai-chat"``.
    - `default_temperature` / `default_reasoning`: stamped onto the provider;
      pass ``None`` for endpoints that reject the field (the Responses API
      rejects temperature). `read_reasoning=True` instead reads the effort from
      the environment.
    - `missing_exc`: factory for the error raised when model/token are missing —
      ``SystemExit`` for CLIs (clean message, no traceback), ``SkipTest`` for the
      live e2e tests so credential-less CI skips rather than fails.
    - `reexport_auth`: when reading the real ``os.environ`` (``env is None``),
      write the stripped token back so the adapter — which reads ``os.environ``
      directly — sees a clean value even if the user's export had stray spaces.

    The Responses API caps output tokens by default
    (`DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS`); other kinds leave it unset.
    """
    source = env if env is not None else os.environ

    resolved_kind = (
        api_kind
        or _pick(source, names.api_kind, fallback.api_kind if fallback else None)[0]
        or "openai-chat"
    )
    if resolved_kind not in API_KIND_CHOICES:
        raise missing_exc(
            f"Unsupported API kind {resolved_kind!r} for {label}: {API_KIND_CHOICES}"
        )

    model, _ = _pick(source, names.model, fallback.model if fallback else None)
    token, auth_env = _pick(source, names.auth, fallback.auth if fallback else None)
    base_url, _ = _pick(source, names.base_url, fallback.base_url if fallback else None)

    missing = [
        env_name
        for env_name, value in ((names.model, model), (auth_env, token))
        if not value
    ]
    if missing:
        raise missing_exc(f"Missing env for {label}: " + ", ".join(missing))

    if reexport_auth and env is None:
        os.environ[auth_env] = token

    reasoning = reasoning_from_env(source) if read_reasoning else default_reasoning
    return Provider(
        id=resolved_kind,
        api=cast(ApiKind, resolved_kind),
        model=model,
        base_url=base_url or None,
        api_key_env=auth_env,
        default_max_tokens=api_kind_defaults(resolved_kind).default_max_tokens,
        default_temperature=default_temperature,
        default_reasoning=reasoning,
        context_window=default_context_window_book().window_for(model),
    )


def resolve_api_key(provider: Provider, *, placeholder: str | None) -> str | None:
    """Resolve a provider's API key from its `api_key_env`, shared by adapters.

    Replaces three near-identical adapter `_api_key` helpers. `placeholder` is
    what to return when no env var is configured (``api_key_env == ""``): OpenAI
    SDKs reject an empty key, so those adapters pass ``"not-needed"``; the
    Anthropic SDK accepts ``None``. A configured-but-unset env var is always a
    hard error so a missing credential fails clearly instead of as a 401.
    """
    env = provider.api_key_env
    if not env:
        return placeholder
    api_key = os.environ.get(env)
    if not api_key:
        raise RuntimeError(
            f"Provider {provider.id!r} requires env var {env!r}; not set."
        )
    return api_key
