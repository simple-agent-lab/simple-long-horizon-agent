---
title: "Bound Filesystem Memory Growth"
status: Accepted
date: 2026-07-14
slug: bound-filesystem-memory-growth
---

# Bound Filesystem Memory Growth

## Context

`FilesystemMemory` originally retained every run directory, complete transcript,
artifact, and `INDEX.md` row forever. Reusing one run id also created `_2`, `_3`,
and later directories. Because the distiller produced a complete `MEMORY.md`, the
collision retargeting step globally replaced the old path inside that complete
file; retries could therefore corrupt historical citations and grow path text
quadratically.

Two existing SWE-bench Pro memory roots contain 261 runs each and occupy about
70 MB and 62 MB. Transcripts account for about 90% of both roots. Their largest
namespace has 31 runs, largest transcript is about 0.94 MiB, and largest
artifact is about 0.28 MiB. A separate vendored chain reaches 56 runs. The
default policy should preserve those ordinary cases while bounding a long-lived
`~/.simple/memory` root.

Codex uses a larger bounded exact set for active memory consolidation and
deletes rollout summaries outside the retained set. Simple Agent Lab needs the
same bounded-set principle without Codex's job database, leases, Git workspace,
or background pipeline.

## Decision

Keep filesystem memory synchronous and inspectable, with limits collected in
one `FilesystemMemoryLimits` value.

Default per-namespace storage limits are:

- 64 retained runs and 128 MiB total namespace size;
- 20,000-byte task, 1,000,000-byte transcript;
- 16 artifacts, 500,000 bytes each and 1,000,000 bytes combined;
- 12,000-character run and navigation summaries, and 2,000-character
  `INDEX.md` cells;
- the existing 20,000-character handbook cap, with at most 32 distinct run
  references.

Root-wide active-cache targets are 1,024 runs, 512 MiB, and 128 cited runs. A
general or reused root admits at most 128 namespace directories by default.
Automatic routing may create 64 learned namespaces, then falls back to
`default`. A caller-provided
`memory_name` is never silently redirected: admitting a new explicit namespace
is refused once 128 directories exist. Admission never evicts a namespace:
without an activity lease, an empty-looking skeleton may belong to a run between
`initial(...)` and `finish(...)`. Existing namespaces remain usable when a
legacy root is already over a newly lowered cap, so consolidation can reduce it;
only count-increasing admission fails closed. Batch admission reserves the whole
requested set with one capacity decision under the root lock. If preparing any
new layout fails, it removes the other new layouts from that batch and leaves
pre-existing namespaces untouched; an unremovable rollback is reported in the
stable warning file.

The SWE memory-chain runner knows its complete finite batch before admission.
For a fresh run-local memory root it therefore records and uses
`max(128, planned namespace count)`; this lets an intentional full-split
`--singleton-memory` arm fit without removing the bound. A reused
`--memory-home` keeps the 128 default and requires an explicit positive
`--memory-max-namespaces` override to raise it. The chosen value is written to
the experiment manifest.

An eval container that sees only its namespaced child can enforce only the
per-namespace limits. The SWE-chain host runner therefore calls root maintenance
before and after each child-isolated batch, using the same shared lock directory
as the containers; the next batch also repairs an interrupted prior batch. In
that isolated view, the 128-reference root limit is a soft target rather than a
hard write gate because sibling handbooks are not visible. Host maintenance
detects an overage and writes one stable warning. Per-run evidence budgets and
namespace admission are hard new-growth bounds. Per-namespace run, byte, and
reference values are retention/rewrite targets: protected legacy overages are
preserved, but a rewrite above the reference target must strictly reduce it.

The distiller first receives at most 500,000 transcript characters and 100,000
artifact characters, then the final rendered prompt is fit to 512 KiB of UTF-8
bytes including task, prior memory, names, descriptions, and wrappers. The
built-in provider distiller lowers this further to 70% of the provider's usable
input budget when `context_window` is known. The final ceiling cannot be lower
than 16,000 bytes; provider construction fails clearly when output and safety
reserves leave less room than that. Dynamic routing sees detailed
context for the eight most recently updated namespaces; the cold-start names
section lists at most 64 names, limits each description to 200 characters, and
is at most 8,000 bytes in total.

Run persistence follows these rules:

1. Bound task, transcript, and artifacts before the model call and disk write.
   Head/tail truncation includes original size and a SHA-256 prefix.
2. Treat the bounded evidence fingerprint as the retry identity. The same run
   id and evidence is a no-op; different evidence under the same id receives a
   deterministic content-hash suffix.
3. Write a `writing` `.pending.json` before evidence, then journal a bounded v2
   `prepared` plan containing the index row, summaries, guarded handbook rewrite,
   and before/proposed handbook hashes. Write `.commit.json` last. Startup
   maintenance removes safe `writing` remnants; it forward-completes a prepared
   plan without a second model call or duplicate lesson. If recovery remains
   blocked, retention pins that pending run and the namespace refuses new
   evidence until recovery succeeds, preventing an error loop from accumulating
   unlimited prepared directories.
4. A distiller can return `retain_run=false`; a successful no-op then creates no
   run evidence or index row. Distiller failures remain inspectable evidence.
5. After commit, prune the oldest runs until both count and byte limits hold.
   Keep the current run and runs cited by `MEMORY.md`, verify that each removal
   actually succeeded, filter `INDEX.md`, and rebuild `memory_summary.md` when
   pruning occurs. A failed removal stays accounted for and produces a stable
   warning; it is never reported as reclaimed space. Because the current run is
   committed before that removal attempt, count retention can temporarily reach
   its target plus one. Later evidence is refused for that namespace or root
   until a maintenance pass really removes the excess, including when the
   warning file itself cannot be written.
   If atomic replacement and cleanup of its hidden temp both fail, create one
   non-atomic `.memory-write-blocked` sentinel in the affected namespace, or a
   shared `.memory-lock/root-write-blocked` sentinel for the full root, and stop
   retry/error writes there. Clear it only after maintenance can inspect that
   same scope and remove every stale temp. A child-only mount must never clear a
   sibling namespace's sentinel or the shared root sentinel; the backend marks
   that view with `SAL_MEMORY_ROOT_VIEW=isolated`.
6. Reject a handbook rewrite that cites missing evidence or newly/nondecreasingly
   exceeds 32 runs. A legacy overage may only move downward. When the writer can
   see the full root, apply the same monotonic guard at 128 pinned runs. Only
   committed runs (plus the current prepared run) and safe evidence paths that
   exist are valid citations.
7. Apply a complete handbook rewrite only when the target namespace was known
   and loaded in full. A one-pass dynamic router may still save the run and
   index row, but it keeps an unseen or truncated target handbook unchanged.
8. Never delete a whole namespace automatically without an activity lease. If
   protected legacy evidence, root citation pressure, or filesystem permissions
   prevent a configured target, keep it and overwrite one stable
   `retention_warning.md` instead of creating unbounded error files. Targets
   blocked only by citations remain warnings; a verified filesystem deletion
   failure additionally stops new evidence so it cannot cause linear growth.
9. Keep local memory private (`0700` directories and `0600` atomic files). A
   root-run Docker worker discovers the host owner from the shared lock bind and
   transfers ownership of new and legacy visible memory back to that owner;
   it does not widen files to world-readable or world-writable modes.
10. Preserve portable namespace names, but encode unsafe, overlong,
    root-reserved, Windows-device, and internal-prefix inputs into an
    80-character maximum component with a 20-hex-character SHA-256 suffix.
    Exact names returned by namespace listing round-trip to that existing
    directory; case-insensitive aliases are refused. The eval runner's `salx-`
    output is a fixed point under the general encoder. Count namespace symlinks
    conservatively but never use them for recall, writes, cleanup, or ownership
    repair. Persist text with LF newlines on every platform so byte accounting
    matches the actual file.

## Consequences

- Ordinary current eval memories remain below the defaults, while long-lived
  roots converge to finite namespace/run/file/byte targets. Legacy data that is
  already over the pinned-reference cap is preserved and reported rather than
  silently deleted.
- Raw evidence is a bounded cache, not an eternal audit log. Important lessons
  must be consolidated into `MEMORY.md`; callers needing complete trajectories
  should use the trace or artifact store.
- Identical successful retries no longer call the distiller or add files.
  Distiller failures are committed as inspectable evidence but retry
  consolidation in place; legitimately different attempts that reuse a caller
  id remain distinct and deterministic.
- UTF-8 byte limits are stricter and more predictable for disk use than
  character-only limits. Preliminary component limits may use characters, but
  the final rendered distiller prompt is authoritatively capped in UTF-8 bytes.
- Unsafe legacy names produced by the former lossy sanitizer are not
  automatically guessed. Normal portable names retain their existing paths;
  ambiguous unsafe names need an explicit one-time migration.
- Retention is deterministic by commit/index order. No TTL is added because the
  starter backend has no trustworthy last-used telemetry.
- This is still not a multi-file transaction. The prepared journal makes the
  bounded intended commit replayable, while before/proposed handbook hashes
  prevent replay from overwriting an unrelated later handbook change.

## Alternatives Considered

- **Keep everything and add only a manual cleanup command.** This leaves the
  default long-lived root unsafe and cannot bound model context.
- **Use only TTL.** Wall-clock age does not express teaching/eval relevance and
  is harder to test deterministically.
- **Reject every reused run id.** That would lose intentional attempts when an
  experiment reuses a shared memory home. Content-addressed suffixes preserve
  those attempts while making exact retries idempotent.
- **Copy Codex's full memory pipeline.** Its database, leases, usage ranking,
  and asynchronous consolidation are unnecessary for this small synchronous
  mechanism.
