---
title: "Consolidate Provider Construction and Env-Var Reading in llm.env"
status: Accepted
date: 2026-06-13
slug: consolidate-provider-env
---

# Consolidate Provider Construction and Env-Var Reading in llm.env

## Context

Building a `Provider` from the environment had drifted into many copies. An
audit found the same `OPENAI_MODEL_ENV = "OPENAI_MODEL"` triple declared in 8+
files; six near-identical `build_*_provider()` / `*_provider_from_env()`
helpers; four byte-for-byte copies of a `load_dotenv()` loader; twelve
hardcoded `Provider(id="fake", api="fake", model="fake-model")` literals; and
three adapter `_api_key()` helpers identical but for one branch. Beyond the
duplication there were genuine inconsistencies: adapter docstrings advertised
`OPENAI_API_KEY` as the default key env while every real call site used
`OPENAI_AUTH_TOKEN` (a gateway bearer token); "api kind" was expressed three
overlapping ways (`Provider.api`, the `API_KIND` env var, and a `kind="openai"`
selector); and nothing tied the wire adapter to the endpoint, so pointing
`OPENAI_BASE_URL` at an Anthropic-protocol URL while `API_KIND` stayed
`openai-chat` surfaced only as a confusing gateway 5xx.

Each copy was an independent chance to drift — exactly how the base-url/adapter
mismatch above went unnoticed.

## Decision

`simple_agent_lab.llm.env` is the single source of truth for building a
`Provider` from the environment. It owns:

- the canonical env-var name constants (`OPENAI_*`, `JUDGE_*`, `API_KIND`,
  reasoning, session/log);
- `ProviderEnvNames` plus the `OPENAI_ENV` / `JUDGE_ENV` bundles, so a primary
  name set and an optional fallback (the judge → `OPENAI_*`) are data, not
  string-prefix guessing;
- one `provider_from_env(...)` covering every former copy, with knobs for the
  real differences (`api_kind`, `default_temperature`/`default_reasoning` or
  `read_reasoning`, `missing_exc` so CLIs raise `SystemExit` and live tests
  raise `SkipTest`, `reexport_auth`);
- one `load_dotenv()`, one `FAKE_PROVIDER`, `reasoning_from_env`,
  `request_extra_from_env`;
- `resolve_api_key(provider, *, placeholder)`, shared by all three adapters
  (the only real difference — OpenAI SDKs reject an empty key and pass a
  placeholder, the Anthropic SDK accepts `None` — is now an explicit argument).

Every other site imports from this module. The framework's existing entry
points (`in_container.provider_from_env(kind=...)`,
`onemillion.judge_provider_from_env()`, the harness env-name constants and
`load_dotenv`) are kept as thin wrappers / re-exports so callers and tests are
undisturbed; the *logic* lives in one place. `Provider.default_temperature`
was widened to `float | None` to match a long-standing real usage (the
Responses API rejects temperature) that only type-checked once it moved into
`src`.

## Consequences

- One place to change an env-var name, the dotenv rules, or the provider
  defaults; the drift that hid the base-url/adapter mismatch can't recur.
- The endpoint/adapter mismatch is now documented at the source (the `llm.env`
  module docstring) rather than rediscovered per incident; a principled
  pre-flight validator was rejected as too endpoint-specific to be reliable.
- A new provider call site is one `provider_from_env(...)` call, not a copied
  helper.
- `llm.env` is imported by adapters, scripts, the gateway, the eval framework,
  and the harnesses — but it depends only on `llm.provider`, so the dependency
  still points inward and there is no cycle.

## Alternatives Considered

- **Leave the framework copy in `src/simple_agent_lab/evals/in_container.py` as
  canonical.** It
  already worked, but it lives in `evals`, so scripts and the gateway could not
  depend on it without an upward dependency — which is exactly why they each
  grew their own copy. The home has to be `llm`.
- **Prefix-string derivation for JUDGE vs OPENAI.** Clean for
  model/auth/base-url but breaks on the `API_KIND` vs `JUDGE_API_KIND`
  irregularity. `ProviderEnvNames` bundles make the irregularity explicit data.
- **A pre-flight validator that rejects an Anthropic base_url on an
  openai-chat provider.** The "/anthropic" heuristic is gateway-specific and
  fragile; documentation plus surfacing the resolved (api, base_url, model) is
  the honest fix.
- **Unify the three adapter `_api_key` helpers into one identical function.**
  Impossible without erasing a real SDK difference (empty-key placeholder vs
  `None`); the shared helper keeps it as one explicit parameter instead.
