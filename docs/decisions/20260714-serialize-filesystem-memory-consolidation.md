---
title: "Serialize Filesystem Memory Consolidation"
status: Accepted
date: 2026-07-14
slug: serialize-filesystem-memory-consolidation
---

# Serialize Filesystem Memory Consolidation

## Context

`FilesystemMemory.finish(...)` is a read-modify-write operation over shared
derived files. It reads `INDEX.md`, `MEMORY.md`, and `memory_summary.md`, asks an
optional distiller for a complete handbook rewrite, then replaces those files.
Atomic replacement prevents readers from observing partial files, but two
processes can still read the same old snapshot and silently overwrite one
another's index or handbook update.

Codex avoids this class of race by allowing parallel per-rollout extraction but
claiming a singleton global job before filesystem consolidation. Simple Agent
Lab needs the same single-writer invariant without adopting Codex's state
database, lease heartbeat, background pipeline, or Git workspace baseline.

The final namespace cannot always be used as the lock key. When callers omit
`MemoryContext.memory_name`, the distiller chooses a namespace only after it has
read existing memories. A namespace lock therefore cannot reliably be acquired
before the stale read without first separating namespace routing from
consolidation.

## Decision

Serialize each `FilesystemMemory` finish operation with one inter-process lock
scoped to its memory root.

- Acquire the lock before reading any existing memory used for distillation.
- Use the same lock for first-time layout creation in `initial(...)`, so an
  `INDEX.md`/`MEMORY.md` skeleton can never replace a concurrent update.
- Hold it through the distiller call, run-directory allocation, evidence write,
  and updates to `INDEX.md`, `MEMORY.md`, and `memory_summary.md`.
- Keep the existing temporary-file plus `os.replace(...)` writes. The lock
  prevents logical lost updates; atomic replacement prevents partial files.
- Use the small cross-platform `filelock` package because the project declares
  OS-independent Python support. The lock remains filesystem-local and requires
  every writer to use `FilesystemMemory` against the same lock-capable shared
  filesystem.
- Keep the implementation in `memory/filesystem.py`; do not add a lock service,
  job database, or background worker.
- When an eval container mounts only one namespace child, also bind-mount the
  shared host `<memory-root>/.memory-lock/` directory at the same root-relative
  path. Mount the directory, not one lock file: `filelock` may unlink/recreate
  `memory.lock`, and every writer must observe that new directory entry. Create
  the directory and private lock file on the host before container start. A
  root-run worker uses its owner as the ownership handoff target for visible
  memory data, so host maintenance can prune it without widening transcript or
  artifact permissions.

## Consequences

- Concurrent runs sharing a memory root no longer distill or commit from stale
  snapshots, including runs that select their namespace dynamically.
- Different namespaces under one root also serialize. The distiller model call
  is inside the critical section, so a slow call delays later finishes on that
  root. This is the deliberate simplicity/correctness tradeoff for the starter
  backend.
- The operating system releases the underlying lock when a process exits. The
  existing distiller timeout bounds ordinary model-call stalls.
- This is not a multi-file transaction. Per-run writing/prepared/commit markers
  let lock-time maintenance remove early partial evidence. The prepared marker
  contains a bounded commit plan plus handbook before/proposed hashes, so the
  next lock holder can forward-complete it without a second model call or stale
  overwrite. There is no background worker; recovery runs synchronously when
  the next recall, finish, or explicit host maintenance acquires the lock.
- Remote object stores and filesystems without reliable inter-process locking
  are outside this backend's coordination guarantee.

## Alternatives Considered

- **Lock each namespace independently.** This would preserve parallelism across
  namespaces, but it is unsafe while the distiller may choose the namespace
  after reading existing memory. Revisit it if routing becomes a separate phase.
- **Lock only individual writes.** This prevents simultaneous replacement but
  still lets both distillers compute full rewrites from the same stale input.
- **Use optimistic compare-and-swap.** A conflict would require another costly
  model distillation and a version/transaction protocol across several files.
- **Copy Codex's SQLite leases and two-phase background pipeline.** It supports
  retries, takeover, and higher throughput, but adds much more machinery than
  the synchronous educational backend needs.
