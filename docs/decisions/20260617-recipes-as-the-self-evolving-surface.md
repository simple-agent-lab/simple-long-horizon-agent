---
title: "Recipes Are The Self-Evolving Surface; Infra Stays Benchmark-Agnostic"
status: Accepted
date: 2026-06-17
slug: recipes-as-the-self-evolving-surface
note: builds on `faithful-hyperagents-recipe` and `retarget-recipe-to-dgm`
---

# Recipes Are The Self-Evolving Surface; Infra Stays Benchmark-Agnostic

## Context

After the DGM retarget (`retarget-recipe-to-dgm`), the self-evolving code had
grown into a sprawl under the old `scripts/` evolution-recipes tree (a
hyperagents and a dgm subtree, two top-level runners, ad-hoc split/report
tools) mixed together
with benchmark glue, Docker probing, and the generic evolution machinery. The
boundary between "the framework" and "one example on one benchmark" was no
longer legible — a researcher (and even the project owner's mentor) could not
tell what to reuse versus what was SWE-bench-specific scaffolding.

The project's mission is an infra that others can read and extend, VeRL-style:
a small benchmark-agnostic core plus a few faithful example recipes. That
requires a durable, enforced boundary, not just tidier files.

## Decision

Adopt a **three-tier** structure with an enforced import boundary.

1. **Substrate — `src/simple_agent_lab/evolution/` (benchmark-agnostic).** The
   kernel (`store`, `log`, `loop`, `experiment`, `types`) owns the guarantees:
   immutable content-addressed versions, fair same-slice A/B, append-only
   decision log, log-then-promote. Swappable components live in `components/`
   (`reward`, `criterion`, `rollout`, `strategy`), alongside `archive.py` and the
   parallel `open_ended.py` loop. Generic patterns belong here: the model-driven
   whole-program meta-strategy (`components/strategy.py`) and the open-ended
   archive-admission loop are benchmark-agnostic and live in the substrate. The
   substrate imports **no benchmark**.

2. **Adapter — `src/simple_agent_lab/evals/suites/swebench/evolving_rollout.py`.**
   All SWE-bench-specific glue (rollout construction, dataset/command helpers,
   grading reuse, performance layout, seed package) lives in the eval suite, next
   to the benchmark it serves. It is **Docker-import-free**: `scripts/arch_lint.py`
   restricts `import docker` to `evals.backends`.

3. **Recipes — `recipes/` (the user surface).** Thin, runnable examples compose
   the tiers: `simple/` (sequential `Experiment.run`, minimal code) and `dgm/`
   (parallel open-ended loop, all knobs) plus small recipe-local ops scripts
   (`dgm/baseline.py`, `dgm/report.py`). The recipe layer is the **only** place
   allowed to touch Docker and host env, via `recipes/_shared.py`; it is not
   arch-linted.

A new benchmark adds an adapter plus a recipe and never edits the substrate.
The legacy evolution-recipes tree under `scripts/` and the old top-level runners
are removed; `runs/` wrappers point at the recipes.

## Consequences

- **Easier:** finding the framework (one `evolution/` package), reading a recipe
  end-to-end, and adding a benchmark (adapter + recipe, substrate untouched).
- **Enforced:** the docker boundary and module zoning are checked by
  `arch_lint.py`, so the benchmark-agnostic claim cannot silently rot. Docker
  probing is structurally confined to `recipes/_shared.py`.
- **Harder / out of scope:** one-off operational tooling (dataset splitting,
  bespoke reports) is intentionally kept as small recipe-local scripts rather
  than promoted into the package; throwaway tools (e.g. the old
  `make_swebench_split`) are not committed.
- **Migration cost:** a one-time deletion of the legacy tree; coverage from the
  deleted modules was re-homed into adapter/recipe tests before removal.

## Alternatives Considered

- **Put SWE-bench glue in an `evolution/swebench/` subpackage.** Rejected: it
  pulls a benchmark into the benchmark-agnostic infra, the exact confusion this
  ADR resolves. The eval suite already owns benchmark specifics.
- **Keep Docker helpers in the package (behind lazy imports).** Rejected:
  `arch_lint` forbids `import docker` outside `evals.backends`, and recipes are
  the natural home for host/Docker probing. Confining it to `recipes/_shared.py`
  keeps the package importable without Docker.
- **Leave the legacy evolution-recipes tree in place.** Rejected: the sprawl
  is the problem; a clear surface requires removing the duplicate/legacy paths,
  not documenting around them.
