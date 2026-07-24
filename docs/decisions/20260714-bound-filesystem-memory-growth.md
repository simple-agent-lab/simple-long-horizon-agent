---
title: "Bound Filesystem Memory Growth"
status: Accepted
date: 2026-07-14
slug: bound-filesystem-memory-growth
---

# Bound Filesystem Memory Growth

## Context

`FilesystemMemory` originally kept every run, transcript, and artifact forever.
Repeated use of a shared `~/.simple/memory` root therefore had no storage bound.
The starter backend needs predictable limits, but it should remain a small,
inspectable Markdown store rather than grow a transaction or recovery protocol.

## Decision

Keep one small `FilesystemMemoryLimits` value with these defaults:

- 128 namespace directories per root;
- 64 run directories and 128 MiB per namespace;
- 20,000 bytes for the task and 1,000,000 bytes for the transcript;
- 16 artifacts, at most 500,000 bytes each and 1,000,000 bytes combined.

The distiller still sees at most 500,000 transcript characters. When it chooses
among existing namespaces, only the first eight namespace summaries, indexes,
and handbooks are loaded. Cold-start recall lists at most 64 namespaces.

Run persistence follows four direct rules:

1. Truncate task, transcript, and artifact content before the model call and
   before writing it to disk.
2. Treat `run_id` as the idempotency key. If that run already has its core
   evidence files, `finish(...)` is a no-op. Callers that need distinct attempts
   must provide distinct run ids.
3. A distiller result with `retain_run=false` creates no durable run. A
   distiller failure still writes bounded evidence and a compact error marker.
4. After a write, delete oldest run directories until the namespace satisfies
   its run-count and byte limits. Remove their `INDEX.md` rows and rebuild
   `memory_summary.md`. The run just written is protected during that pass.

Namespace admission only checks the finite directory count. Existing namespaces
are never evicted automatically. Unsafe or overlong logical names receive a
short hash suffix so different path-like names do not collapse onto the same
ordinary sanitized component. The SWE memory-chain runner may raise the
namespace count for a finite run-local plan and records that choice in its
manifest.

All read-distill-write work remains under the existing root-scoped file lock,
and each individual file is replaced atomically. Docker child mounts continue
to share the root lock directory. A root container gives newly created entries
back to the host lock owner when the platform supports ownership changes.

This backend deliberately has no prepared/commit journal, write-block sentinel,
root-wide citation accounting, automatic namespace deletion, or automatic
multi-file crash recovery. A process failure can leave a partial run; a later
retry may replace that incomplete directory. Persistent filesystem failures are
reported through one stable root `memory_error.md` and do not fail the agent
run.

## Consequences

- Growth is finite and ordinary long-lived roots remain easy to inspect.
- The implementation stays synchronous and small enough to teach directly.
- Retention is a cache policy, so old raw evidence can disappear even when a
  handbook still mentions it. Durable lessons should remain understandable
  without requiring every historical transcript.
- Same-id retries trade perfect attempt preservation for simple idempotency.
- This is not a transactional store. Users needing complete audit history or
  automatic crash recovery should use the trace/artifact system or another
  backend.

## Alternatives Considered

- **Keep every run.** Simple, but unsafe for a long-lived default root.
- **Add a journal and recovery state machine.** More durable, but much larger
  than the educational filesystem backend warrants.
- **Copy a production memory pipeline.** Databases, leases, background jobs, and
  ownership repair are outside this starter mechanism.
