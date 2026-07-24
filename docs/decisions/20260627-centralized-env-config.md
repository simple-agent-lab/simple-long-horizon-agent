---
title: "Centralized Env-Config Registry"
status: Proposed
date: 2026-06-27
slug: centralized-env-config
---

# Centralized Env-Config Registry

## Context

Configuration env vars are scattered: ~37 `os.environ` reads across ~16
modules, ~35 `*_ENV = "..."` name constants defined in a dozen places, and
each read re-implements its own "read + default + coerce" logic. Some names
are duplicated (`SAL_AGENT_COMPRESSION_*` is declared in both
`agent_flavors.py` and `agents/flavors.py`), and the SWE-bench workflow knobs
travel through an indirection chain: `src/simple_agent_lab/agents/flavors.py`
defines them, `src/simple_agent_lab/evals/suites/swebench/container.py`
re-exports them, and the run script + tests import them from there. The only "central" view is
`docs/agent-native/configuration.md`, a hand-maintained catalog that drifts
(we recently fixed broken code-path references in it by hand).

The run-profile precedence bug (a profile's `store_true` / `append` flags do
not follow the documented "an explicit CLI flag overrides" rule) is a symptom
of this: there are two precedence models — `env` fill-the-gaps and a `run`
section that piggybacks on argparse token order — and the second is leaky.

Partial precedents already exist: `llm/env.py` consolidated all provider env
(ADR consolidate-provider-env), and `agent_flavors` is a FOUNDATION-zone leaf
of shared config name constants. This ADR generalizes that pattern to the
whole package.

## Decision

Introduce one declarative config registry — `simple_agent_lab/config.py`, a
FOUNDATION-zone leaf with no internal deps so every layer can import it. Each
env-backed value is declared once as an `EnvVar(name, default, group, doc,
parse)`:

- `group` is a dotted `domain.subsystem` label — the **config hierarchy**.
  Top-level domains are `agent`, `eval`, `provider`, `trace`, `runtime`.
  Classification is by domain, **not** by an env var's name prefix. Because the
  registry centralizes the name, a historical misnomer is now a one-line fix:
  the workflow-arm knobs were renamed from `SWE_*` to `SAL_WORKFLOW_*` (they
  configure the *agent's* workflow arm — `agent.workflow` — not SWE-bench),
  while `SWE_REPO_LANGUAGE` keeps its prefix because it really is eval-container
  config (`eval.swebench`).
- `EnvVar.get()` applies **one precedence rule**: an explicit override beats
  the environment, which beats the declared default; a blank/unparseable value
  falls back to the default and never raises (matching the ad-hoc readers it
  replaces).

Consequences for the precedence bug: a profile and a CLI flag both become
*override layers* fed to the same resolver, so there is one uniform rule and
no argv-token tricks. `configuration.md` becomes generated-from / validated-
against `REGISTRY`, so it cannot drift.

Migration is incremental. This ADR ships the registry plus a sample migration
of the SWE-bench env vars (both hierarchy levels), including the `SWE_*` ->
`SAL_WORKFLOW_*` rename of the generic workflow knobs (a breaking env-name
change, acceptable for these alpha research knobs). To keep each step
reviewable and low-risk, a consumer migrates its *reads* to `config.X.get()`
first; a name constant that other modules still import is kept as a thin
transitional alias (`PDR_ROUNDS_ENV = config.PDR_ROUNDS.name`) until those
callers are migrated too, after which the alias and any re-export are removed.

The registry stays a flat, explicit list of small `EnvVar` declarations — no
reflection, no auto-discovery, no dynamic magic (per AGENTS.md: small explicit
modules, no heavy framework, no magic configuration).

## Consequences

- One place to see and manage every config knob; type, default, and docs live
  in the declaration.
- One precedence model end to end, which dissolves the run-profile
  `store_true` / `append` override bug.
- Removes duplicated name constants and the SWE-bench re-export indirection.
- `configuration.md` can be generated/validated from the registry (no drift).
- Costs: one new module, and an incremental migration that touches many
  files over time. During migration, transitional aliases mean a name can
  briefly live in two spellings (the registry + an alias), which is the
  explicit, documented trade-off for keeping each step small.

## Alternatives Considered

- **Names + docs only**: a constants module plus a generated catalog. Centralizes
  names but leaves the per-read coercion scatter and the precedence bug.
- **Collapse the profile `run` section into per-flag env defaults**: uniform, but
  turns CLI knobs into env vars and inherits argparse's `append`/`default`
  gotchas.
- **Patch the profile precedence in place** (drop profile keys the CLI sets):
  fixes the symptom, not the underlying scatter.
