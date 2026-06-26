---
title: "JSON Run-Profile File for Launching an Agent + Bench"
status: Accepted
date: 2026-06-26
slug: run-profile-file
---

# JSON Run-Profile File for Launching an Agent + Bench

## Context

Launching one agent-on-a-bench arm reads from two surfaces: `.env` (the provider
/ flavor / workflow / compression env knobs, forwarded into the container by each
harness `container_environment()` / passthrough list) and the `run_*_suite.py`
argparse flags (run-shape: instance, max-turns, image, cpus/memory, network,
dataset). The `run_swebench_arms.sh` driver glues both together in bash. An arm
is therefore real but implicit — to reproduce "PDR on SWE-bench Pro at 200
turns" you must remember a flavor env var, a compression ratio, and a handful of
flags, spread across a shell history and a `.env`.

Operators asked for the mini-swe-agent ergonomic: one committed file that names a
runnable arm, instead of reconstructing it from scattered exports and flags.

The env-var *names* are already consolidated (ADR `consolidate-provider-env`) and
catalogued (`docs/agent-native/configuration.md`); this is about the *launch
bundle*, not the names.

## Decision

Add a single JSON **run-profile** file with two sections and a small loader
(`simple_agent_lab.evals.profile`), wired into the run entry points behind one
`--profile PATH` flag:

```json
{
  "_comment": "PDR arm on SWE-bench Pro",
  "env": { "AGENT_FLAVOR": "pdr", "SWE_PDR_WIDTH": "3" },
  "run": { "max-turns": 200, "network-mode": "host", "prepare-wheelhouse": true }
}
```

- `env` → environment variables. Applied **fill-the-gaps**, identical to
  `load_dotenv`: a value already exported (or set by a later `.env`) wins, so the
  profile is the document and an ad-hoc export is still the override. The keys are
  the same canonical names catalogued in `configuration.md`.
- `run` → run-shape knobs whose keys are the `run_*_suite.py` long-option names
  (without `--`). They are expanded to argv and injected *before* the real
  command line, so an explicit CLI flag overrides the profile. The loader does
  not know any suite's flag set; argparse validates run keys (an unknown one is a
  normal argparse error), keeping the loader suite-agnostic.

**JSON, not YAML or TOML.** This mirrors ADR `model-config-file` exactly: YAML is
a new hard dependency against the project's two-dep minimalism, and `tomllib` is
stdlib only on 3.11+ while the floor is 3.10. A leading-`_` key is ignored, so
JSON gets the one thing it lacks — a comment channel — and the example file
documents itself.

**One precedence rule, no magic.** Both sections degrade to "the profile is a
default": real env wins over `env`, the real command line wins over `run`. There
is no fourth layer and no per-key precedence table — the thing AGENTS.md warns
against ("do not hide core behavior behind magic configuration").

**Not a second source of truth.** The profile carries no new schema for the knobs
themselves: `env` is the catalogued env names, `run` is the existing CLI. It is a
*bundle* of the two existing surfaces, so there is nothing to drift out of sync —
unlike a parallel settings schema would be.

Secrets stay out of committed profiles: provider tokens belong in `.env` /
exports, not a profile under version control. `runs/profiles/*.json` is
gitignored except `*.example.json`, mirroring the `models.json` /
`models.example.json` convention.

The flag is opt-in and additive: every existing `.env` + CLI invocation keeps
working unchanged; `--profile` only pre-seeds the two surfaces.

## Consequences

- A runnable arm is one committed, reviewable file (`runs/profiles/<arm>.json`);
  `run_swebench_arms.sh` arms can point at profiles instead of re-encoding flag
  strings.
- No new dependency, no new schema for the knobs, no precedence table: the
  profile is a thin bundle over `.env` (fill-gaps) and the CLI (overridable
  prefix).
- The loader is suite-agnostic (env map + run map), so adding `--profile` to a
  new suite's run entry is a few lines; argparse remains the authority on which
  run flags exist.
- Two ways to launch now exist (raw `.env`+flags, or `--profile`), but they are
  the same two surfaces, not two schemas — the documented precedence makes the
  combination predictable.

## Alternatives Considered

- **YAML profile (as the operator sketched).** Nicer hand-editing, but a new hard
  dependency; rejected for the same reason as ADR `model-config-file`. The
  `_`-comment convention recovers JSON's missing comments.
- **A full settings schema (one document owning every knob with its own
  precedence stack).** This is the "magic configuration" AGENTS.md forbids and a
  second source of truth beside `.env`/CLI that would drift. Rejected for a thin
  bundle of the existing two surfaces.
- **Bundle only env, leave run-shape on the CLI.** Simpler, but the arm is still
  split across a file and remembered flags; the `run` section is what makes the
  file a complete, reproducible arm.
- **Profile env overrides real exports (profile wins).** Rejected: fill-gaps
  matches `.env` and preserves the ad-hoc export as the always-available
  override, which operators rely on for one-off tweaks.
