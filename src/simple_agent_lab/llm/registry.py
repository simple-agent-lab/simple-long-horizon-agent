"""Named model aliases — a lightweight registry over `provider_from_env`.

Lets a deployment configure a few model *roles* by alias (``strong``, ``fast``,
...) and look them up, instead of every call site hardcoding one env contract.
A strong model for the main agent plus a cheap ``fast`` one for compaction or
sub-agents is the motivating case.

Env contract: each alias reads ``<ALIAS>_MODEL`` / ``<ALIAS>_AUTH_TOKEN`` /
``<ALIAS>_BASE_URL`` / ``<ALIAS>_API_KIND``, falling back to the base
``OPENAI_*`` set (see `simple_agent_lab.llm.env`). So the minimal setup — just
``OPENAI_MODEL`` + ``OPENAI_AUTH_TOKEN`` — resolves *every* alias to that one
model, and existing single-model deployments keep working unchanged. Set
``FAST_MODEL=...`` to peel ``fast`` onto a cheaper model while still sharing the
gateway token and base url from ``OPENAI_*``::

    # OPENAI_MODEL=big OPENAI_AUTH_TOKEN=...   FAST_MODEL=small
    registry = ModelRegistry.from_env()
    main = make_llm_agent(name="main", provider=registry.get("strong"))
    summarizer = make_llm_agent(name="sum", provider=registry.get("fast"))

For a per-alias setting the env scheme can't express (e.g. a different
temperature), build the providers yourself and pass them in:
``ModelRegistry({"strong": p1, "fast": p2})``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .env import OPENAI_ENV, ProviderEnvNames, provider_from_env
from .provider import Provider

# The two roles the project ships by default. Callers can pass any aliases to
# `ModelRegistry.from_env`; these are just the well-known names.
STRONG_ALIAS = "strong"
FAST_ALIAS = "fast"
DEFAULT_MODEL_ALIASES: tuple[str, ...] = (STRONG_ALIAS, FAST_ALIAS)


def env_names_for(alias: str) -> ProviderEnvNames:
    """The four env-var names an alias reads: ``<ALIAS>_MODEL`` etc.

    Always uppercased, so ``"fast"`` reads ``FAST_MODEL`` / ``FAST_AUTH_TOKEN`` /
    ``FAST_BASE_URL`` / ``FAST_API_KIND``. (The base ``OPENAI_*`` set keeps its
    own irregular ``API_KIND`` name via `simple_agent_lab.llm.env.OPENAI_ENV`;
    this regular scheme is for aliases only.)
    """
    prefix = alias.upper()
    return ProviderEnvNames(
        model=f"{prefix}_MODEL",
        auth=f"{prefix}_AUTH_TOKEN",
        base_url=f"{prefix}_BASE_URL",
        api_kind=f"{prefix}_API_KIND",
    )


@dataclass(frozen=True)
class ModelRegistry:
    """An immutable alias → `Provider` map with a strict `get`.

    Build it from the environment with `from_env`, or pass a ready dict for
    programmatic / per-alias control. Lookups are strict: an unknown alias is a
    `KeyError` naming the configured ones, so a typo fails clearly instead of
    silently using the wrong model.
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
    def from_env(
        cls,
        aliases: tuple[str, ...] = DEFAULT_MODEL_ALIASES,
        *,
        env: Mapping[str, str] | None = None,
        fallback: ProviderEnvNames | None = OPENAI_ENV,
        default_temperature: float | None = 1.0,
        read_reasoning: bool = True,
        missing_exc: Callable[[str], BaseException] = SystemExit,
    ) -> "ModelRegistry":
        """Build one `Provider` per alias from ``<ALIAS>_*`` env (fallback OPENAI_*).

        Eager: every alias is resolved now, so a misconfig fails at startup, not
        at first use. Because each alias falls back to ``OPENAI_*``, a base setup
        (``OPENAI_MODEL`` + ``OPENAI_AUTH_TOKEN``) resolves them all to that one
        model; set ``fallback=None`` to require each alias be configured
        explicitly. `read_reasoning` honors the shared ``REASONING_EFFORT`` env
        for every alias (matching how the main agent provider is built).
        """
        providers = {
            alias: provider_from_env(
                env_names_for(alias),
                fallback=fallback,
                env=env,
                default_temperature=default_temperature,
                read_reasoning=read_reasoning,
                label=f"model alias {alias!r}",
                missing_exc=missing_exc,
            )
            for alias in aliases
        }
        return cls(providers)
