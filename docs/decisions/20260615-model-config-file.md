---
title: "JSON Model-Config File for the Model Registry"
status: Accepted
date: 2026-06-15
slug: model-config-file
note: replaces the env-alias scheme of model-alias-registry
---

# JSON Model-Config File for the Model Registry

## Context

`model-alias-registry` (same unmerged branch) added `ModelRegistry` and a
`from_env` that read each alias from `<ALIAS>_MODEL` / `<ALIAS>_AUTH_TOKEN` /
`<ALIAS>_BASE_URL` / `<ALIAS>_API_KIND` with a fallback to the base `OPENAI_*`
set. That ADR itself flagged the file as the eventual home: *"no config-file
(`models.json`) loader yet … the same `ModelRegistry` is the place to add one
later if a deployment outgrows env."*

In review we decided the two surfaces shouldn't both ship. The per-alias env
scheme is a branch-only invention (not on `main`), and two or three models is
exactly where it stops reading well: a handful of roles, each with four
`<ALIAS>_*` vars, is a wall of exported strings with no structure showing which
alias overrides what. A deployment declaring several models wants a *document*,
not a flat namespace. So the file becomes the one multi-model surface and the
`<ALIAS>_*` scheme is dropped before it ever reaches `main`.

## Decision

Add `simple_agent_lab.llm.config` — a JSON loader — plus `ModelRegistry.from_file`
and a `ModelRegistry.load` front door, and **remove** `ModelRegistry.from_env` /
`env_names_for` / the `<ALIAS>_*` env scheme. JSON, not YAML, so the dependency
set stays at two (`anthropic`, `openai`); `Provider` is already JSON-serializable,
and `tomllib` is not available on the supported 3.10 floor.

Amendment (2026-06-15): also accept the same JSON document from
`MODEL_CONFIG_JSON`. This is a transport option for cloud sandboxes where
creating a separate `models.json` is awkward, not a second schema.

Only two configuration modes remain, both `main`-compatible:

- **Multi-model**: the JSON document. It can live in a file
  (`ModelRegistry.from_file(path)` / `MODEL_CONFIG`) or directly in
  `MODEL_CONFIG_JSON`.
- **Single-model**: the base `OPENAI_*` provider `main` already uses.
  `ModelRegistry.load()` returns the JSON registry when `MODEL_CONFIG_JSON` or
  `MODEL_CONFIG` is set, else maps every default alias (`strong`/`fast`) onto
  that one `OPENAI_*` provider — so code asking for a role still works on a
  single-model deployment with no new env.

Schema: a shared `defaults` object and an `aliases` map; each alias merges over
`defaults`. Per-alias keys mirror `Provider`/env semantics (`model`, `api_kind`,
`base_url`, `auth_token_env`, `auth_token`, `reasoning_effort`, `temperature`,
`max_tokens`, `context_window`, `replay_reasoning`). Aliases are whatever the
document names — no fixed `strong`/`fast`. Validation is strict: unknown
top-level or per-alias keys, a bad `api_kind`/`reasoning_effort`, a missing
`model`, or both auth forms on one alias raise `SystemExit` (clean CLI message).
A leading-`_` key is ignored, giving JSON a comment channel so the example file
documents itself.

Secrets: prefer `auth_token_env`, which names the env var holding the token — the
file carries no secret and is safe to commit. An inline `auth_token` is permitted
for quick local runs but is a footgun: loading one emits a `UserWarning` and
stashes the value in a private process env var (`_SAL_MODELCONFIG_<ALIAS>_AUTH_TOKEN`)
so `Provider.api_key_env` and the adapter's existing `os.environ` lookup are
unchanged. `models.json` and `*.local.json` are gitignored; only
`models.example.json` is committed.

`ModelRegistry.load` reads `MODEL_CONFIG_JSON` first, then `MODEL_CONFIG`: set
the former to the JSON text in file-hostile cloud sandboxes, set the latter to a
path for normal file-based config, or leave both unset to run the single
`OPENAI_*` model under every role. So a call site writes `ModelRegistry.load()`
and the deployment picks the mechanism.

`from_file` is eager — every alias resolves at load, so a bad spec fails at
startup, not at first use. The registry stays opt-in: no existing call site is
rewired to require it.

## Consequences

- A multi-model deployment declares its roles in one readable file and asks for
  them by name (`registry.get("fast")`); a single-model one keeps running on
  `OPENAI_*` with no new config, exactly as on `main`.
- One multi-model schema, not two. The `<ALIAS>_*` env scheme is gone, so there
  is no second, drifting way to declare the same roles. `MODEL_CONFIG_JSON` only
  changes how the same JSON document reaches the process.
- The JSON document covers the same `Provider` knobs the env path did plus a few
  literals; anything finer still drops to the programmatic
  `ModelRegistry({...})`.
- Inline tokens are a deliberate footgun, mitigated (warning + gitignore +
  preferred `auth_token_env`) rather than forbidden, so a one-off local run
  doesn't require touching the environment.

## Alternatives Considered

- **Keep the `<ALIAS>_*` env scheme alongside the file.** Rejected: two surfaces
  for the same roles invite drift, and the env one is the worse-reading of the
  two past ~two models. It never shipped to `main`, so there is nothing to keep
  compatible.
- **YAML.** Comments and nicer hand-editing, but a new hard dependency against
  the project's two-dep minimalism. The `_`-prefixed-key convention recovers the
  one thing JSON lacks (comments).
- **TOML (`tomllib`).** Stdlib only on 3.11+; the project supports 3.10, so it
  would need the `tomli` backport — a dependency after all.
- **Inline tokens forbidden, env-ref only.** Safer, but the file's draw for local
  use is being self-contained, so a guarded inline option is worth the footgun.
- **A new env var per inline token vs. extending `Provider` with a literal-key
  field.** Keeping `Provider` pure-data (key *name*, never the secret) preserved
  the adapter contract and JSON-serializability; the synthetic env var is the
  smaller change.
