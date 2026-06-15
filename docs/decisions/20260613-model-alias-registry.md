---
title: "Model-Alias Registry (strong / fast) over provider_from_env"
status: Accepted
date: 2026-06-13
slug: model-alias-registry
note: builds on consolidate-provider-env
---

# Model-Alias Registry (strong / fast) over provider_from_env

## Context

The runtime already supports running several models in one program — `Provider`
is immutable data, nothing is global, and each `Agent` (main, sub-agent,
`SummarizeStrategy` compressor, eval judge) carries its own. What was missing
was a *config* surface: a deployment could not declare a few model roles by name
and look them up. The only second-model pattern was the hardcoded `JUDGE_*` env
set; differentiating "a strong model for the main agent, a cheap fast one for
compaction/sub-agents" meant either inventing another ad-hoc prefix or building
`Provider`s by hand at each call site.

The `consolidate-provider-env` refactor made the mechanism reusable —
`ProviderEnvNames` + `provider_from_env(names, fallback=...)` already builds a
provider from any env-name bundle — so what remained was a thin named lookup
over it.

## Decision

Add `simple_agent_lab.llm.registry` with a small `ModelRegistry` (an immutable
alias → `Provider` map with a strict `get`) and `ModelRegistry.from_env`.

- Aliases default to `("strong", "fast")` but are caller-supplied.
- Each alias reads `<ALIAS>_MODEL` / `<ALIAS>_AUTH_TOKEN` / `<ALIAS>_BASE_URL` /
  `<ALIAS>_API_KIND`, **falling back to the base `OPENAI_*` set**. So a minimal
  `OPENAI_MODEL` + `OPENAI_AUTH_TOKEN` resolves *every* alias to that one model
  — existing single-model deployments keep working untouched — and setting
  `FAST_MODEL=...` peels `fast` onto a cheaper model while still sharing the
  gateway token and base url. `fallback=None` requires each alias be configured
  explicitly.
- `from_env` is eager: every requested alias resolves at construction, so a
  misconfig fails at startup, not at first use. The shared `REASONING_EFFORT`
  env applies to each alias (matching how the main agent provider is built).
- For a per-alias setting the env scheme can't express (e.g. a different
  temperature), callers build the providers and pass `ModelRegistry({...})`.

The registry is opt-in infrastructure: it is exported from `simple_agent_lab.llm`
but no existing call site is rewired to require it.

## Consequences

- A deployment configures `strong`/`fast` (or any aliases) from env and code
  asks for them by name: `make_llm_agent(provider=registry.get("fast"))` for a
  cheap compactor sub-agent, `registry.get("strong")` for the main agent.
- Single-model setups are unaffected — aliases collapse onto `OPENAI_*` by
  fallback, so adopting the registry costs no new env.
- The registry only covers the env-expressible knobs (model/auth/base-url/
  api-kind/reasoning); anything finer drops to the programmatic constructor.
- Still env-first: no config-file (`models.json`) loader yet. The same
  `ModelRegistry` is the place to add one later if a deployment outgrows env.
- The convenience layer (`agent_session`/`make_agent`) still threads one
  provider to its explorer sub-agent; wiring a registry through it (a
  per-sub-agent provider knob) is a separate, still-open follow-up.

## Alternatives Considered

- **A bare `dict[str, Provider]` / a `providers_from_env()` function.** Lighter,
  but the dataclass adds a strict `get` (a typo'd alias names the configured
  ones instead of `KeyError: 'fst'`) and a home for a future file loader, for a
  few lines.
- **Lazy per-alias resolution.** With the `OPENAI_*` fallback, eager
  construction succeeds whenever the base is configured, so laziness buys
  nothing and loses startup validation.
- **A `models.json` config file now.** Heavier than asked; env matches the
  project's existing provider convention. Deferred behind the same type.
- **Reusing `JUDGE_*` as the second slot.** Judge is a distinct eval role, not
  a general alias; overloading it would conflate grading config with model
  identity.
