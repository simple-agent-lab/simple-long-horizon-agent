---
title: "Model-Owned MEMORY.md Handbook Rewrite With Guardrails"
status: Accepted
date: 2026-06-11
slug: model-owned-memory-handbook-rewrite
---

# Model-Owned MEMORY.md Handbook Rewrite With Guardrails

## Status

Accepted

## Context

`FilesystemMemory` maintained `MEMORY.md` with deterministic code. The distiller
returned only an incremental block of bullets (`memory_updates`); `finish(...)`
appended it under a `## Run Updates` section, then a consolidation pass promoted
high-signal bullets into `## Durable Lessons`, normalized and deduplicated keys,
capped the durable section to 40 bullets, and pruned old run-update blocks,
leaving a pointer trail.

That machinery (~250 lines of regex section surgery, a promotion/dedup pipeline,
a sentinel pointer line, and a bullet cap) was the heaviest and least
beginner-readable part of the memory module. It existed to keep the handbook
bounded and non-redundant cheaply, and to keep each run's model output small.
But its dedup is only string-level — it cannot recognize that a new lesson
supersedes or subsumes an older one — and the project explicitly prefers small,
explicit, beginner-readable modules over clever deterministic bookkeeping.

The owner asked whether a simpler model-owned rewrite of `MEMORY.md` would be a
better fit: feed the current handbook to the distiller and let it return the
updated handbook, deciding merges, rewrites, and deletions itself.

## Decision

Make `MEMORY.md` a single model-owned handbook.

- The distiller is shown the current `MEMORY.md` and returns the **complete**
  updated file in a `memory_md` field (replacing the incremental
  `memory_updates`). The model owns all merge, rewrite, reorder, dedup, and
  delete decisions; the prompt asks it to keep the file small (~40 high-signal
  bullets) and evidence-cited.
- `finish(...)` writes that rewrite verbatim. The only remaining deterministic
  logic is guardrails against catastrophic outputs (the model owns content):
  - empty `memory_md` is a no-op — the prior handbook is kept;
  - a rewrite is **rejected** (prior handbook kept, a `memory_error.md` marker
    records the skip) when it is oversized (> `DEFAULT_MAX_HANDBOOK_CHARS`,
    20000), structurally empty (no Markdown heading and no bullet — looks
    truncated), or would drop every bullet from a handbook that had at least
    three (almost always accidental loss, not intentional pruning).
- Removed: the `## Durable Lessons` / `## Run Updates` two-section contract,
  bullet promotion, key-based deduplication, the 40-bullet durable cap,
  run-update pruning/pointers, and their ~250 lines of helpers. The empty
  skeleton is now a single `## Lessons` section, and the run-start policy block
  describes one small curated handbook.

## Consequences

- Easier: far less code and no regex section bookkeeping; the model can perform
  semantic merges and rewrites the old string-level dedup could not; `MEMORY.md`
  has a single owner and source of truth; the module is more beginner-readable.
- Accepted costs/risks: every distillation re-emits the whole handbook, so output
  tokens and latency grow with handbook size; a truncated or runaway model
  response is now possible, mitigated by guardrails that keep the prior handbook
  instead of corrupting or erasing it; growth is now bounded softly (the prompt
  requests ~40 bullets) plus a hard character cap, rather than a deterministic
  bullet cap; a rewrite's content is non-deterministic and not unit-testable —
  only the guardrails and the verbatim-write behavior are.
- Out of scope: semantic-diff verification of rewrites and enforced per-bullet
  provenance (the prompt still asks for greppable evidence anchors, but it is not
  machine-checked).

## Alternatives Considered

- Keep the deterministic two-section consolidation (status quo): bounded, cheap,
  idempotent, and unit-testable, but the heaviest/least-readable code in the
  module and only string-level dedup.
- Single-section incremental: the model returns only new bullets and code
  appends, exact-dedups, and caps (no two-section split). Simpler than the status
  quo while keeping the incremental approach's low output cost and small blast
  radius, but it still cannot do semantic merge and loses the recent-runs trail.
  Rejected because the owner wanted model-driven semantic merging.
- Model-owned full rewrite with no guardrails: simplest of all, but a truncated
  or oversized response could erase or corrupt the entire handbook in one bad
  run. Rejected in favor of the guarded variant above.
