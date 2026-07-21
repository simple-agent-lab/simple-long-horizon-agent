# Memory Sketch

Read when:

- A task proposes memory, recall, persistent context, learned preferences,
  session summaries, workspace notes, or cross-run lessons.
- A task wants to add hooks, memory tools, memory-backed context injection, or
  a storage backend for remembered facts.

Do not read for:

- Ordinary context compression work that only changes which existing transcript
  messages stay visible.
- One-off prompt wording changes that do not introduce persistent or
  cross-turn memory behavior.

## Status

This is an implementation guide for the small memory boundary in
`src/simple_agent_lab/memory/`. It is not an ADR; if the project later commits
to a harder-to-reverse memory architecture, promote that decision into
`docs/decisions/`.

## Goal

Memory should be a small boundary around the current runtime:

```text
Agent + Message + State + build_context_view() + run()
          ^
          |
   memory layer
```

The first design goal is to support different memory methods without choosing a
specific method too early. A file-backed memory, summary memory, vector index,
custom memory tool, prompt injection strategy, or hook-driven learner should
all fit behind the same high-level shape.

## Principles

- Keep `State.events` and `State.messages` as the source of truth for the
  current run.
- Do not change the `Message` protocol just to add memory. Recalled memory
  should become ordinary `Message` values when it enters model-visible context.
- Keep provider adapters memory-agnostic. Provider wire shape should not know
  whether a message came from memory, a user, a tool, or compression.
- Treat memory tools as ordinary `AgentTool` values.
- Make memory actions traceable. Future agents should be able to tell what was
  recalled, injected, written, skipped, or blocked.
- Prefer read-only or explicit-write memory first. Silent long-term writes make
  debugging and teaching harder.
- Current workspace facts, tests, and tool observations outrank stale memory.
- Treat "no memory update" as a successful outcome. Durable memory should
  change future agent behavior, save future user effort, or prevent a known
  failure; otherwise leave it out.
- Prefer progressive disclosure over bulk recall: load a compact summary or
  index first, then search/open detailed memory only when the task needs it.
- Do not start with vector or hybrid RAG as the default. Coding-agent memory is
  usually sparse, high-value, and drift-prone enough that inspectable
  file-backed memory is the better first teaching path.

## Reference Guidance

Gitignored reference material lives under `docs/reference-architectures/` (only
its README and template are tracked). It holds the extracted text of a survey
comparing mainstream agent-harness memory designs (the original survey PDF was
removed), plus local Codex and Hermes source clones, all of which inform the
current direction.

For every future Simple Agent Lab memory optimization, use three inputs
together:

1. The current project code and agent-native docs, with a strong bias toward
   keeping the implementation small, explicit, and beginner-readable.
2. The extracted harness-memory survey text plus the local Codex and Hermes
   source clones. Treat the Claude Code, Codex, Hermes, and other surveyed
   harness directions as comparable reference signals rather than letting one
   product dominate by default.
3. The existing Simple Agent Lab memory implementation and its tests, so
   refinements build on the current teaching shape instead of replacing it
   wholesale.

The owner preference for starter memory mechanisms is one Python file per
mechanism. Keep the active filesystem implementation in `filesystem.py` unless a
split becomes clearly more readable than the extra indirection.

The durable lesson is not to maximize recall. The durable lesson is to control
memory quality:

- Do not save secrets, large raw outputs, generic advice, current-task state,
  or facts that are already cheaper to read from code, docs, or git.
- User messages, explicit corrections, accepted workflows, and tool evidence
  are stronger memory evidence than assistant-authored summaries.
- Memory should carry provenance or evidence pointers when it affects future
  behavior.
- Memory that names current files, functions, flags, commands, or repo state is
  a dated observation. Verify it against the current checkout when verification
  is cheap.
- A fast path can capture explicit user-requested notes or small durable facts;
  a slow path should consolidate, deduplicate, prune, and resolve conflicts.
- Keep human-owned project rules in human-owned docs or ADRs. Automatic memory
  may help recall those rules, but should not be their only home.

## Implemented Shape

The current memory interface is intentionally small, but named as a concrete
thing future users can implement:

```text
Memory
  initial(ctx)        -> tuple[Message, ...]
  recall(ctx, query)  -> tuple[Message, ...]
  tools(ctx)          -> tuple[AgentTool, ...]
  record(ctx, msgs)   -> None
  finish(ctx)         -> None
```

`Memory` is a concrete base class with no-op defaults, not a protocol that
forces every implementation to define every method. Subclass it and override
only the pieces you need. This keeps method-specific choices at the edge while
making the lifecycle explicit. A memory implementation may inject initial
context only, recall context before model requests, expose tools only, observe
completed turns, learn at run end only, or combine these. `MemoryContext`
carries `agent`, `task`, optional `session_id`, optional `run_id`, optional
`memory_name`, optional `step_index`, optional `state`, and extra metadata.

```python
class VectorMemory(Memory):
    def recall(self, ctx, query):
        snippets = self.index.search(query)
        if not snippets:
            return ()
        return (memory_context_message(render(snippets), target=ctx.agent),)
```

Bind memory before the agent is built, then pass its tools through the same
factory path MCP uses. `MemoryBinding.hooks` is a normal core `HookMap`; pass
it to the agent at construction time so the runtime can execute the supported
memory lifecycle points.

```python
binding = memory.bind(
    MemoryContext(
        agent="agent",
        task=task,
        run_id=run_id,
        session_id=session_id,
    )
)
agent = make_llm_agent(
    name="agent",
    provider=provider,
    tools=[*base_tools, *binding.tools],
    hooks=binding.hooks,
)
state, events = agent.run(task)
```

Current core hooks map `initial(...)` to `SESSION_START` and `finish(...)` to
`SESSION_END`. `recall(...)` and `record(...)` remain optional methods for a
future pre-model or after-turn hook point; do not pretend they run live until
the runtime exposes those points.

## Capability Roles

Use small role names when a concrete implementation needs them:

- `MemorySource`: recalls candidate memory for a task, agent, or context view.
- `MemoryLearner`: reads a finished or in-progress `State` and proposes memory
  items worth keeping.
- `MemorySink`: commits approved memory items to a durable place.
- `MemoryToolProvider`: exposes one or more ordinary `AgentTool` values for
  model-directed search or writes.
- `MemoryInjector`: turns recalled memory into runtime `Message` values.

These roles are lower-level parts. The user-facing entry point should stay
`Memory` so later implementations do not have to pretend every memory method is
a store.

## Lifecycle Boundary

The interface deliberately avoids a large hook surface. Initial read happens
through `initial(...)`, query-shaped recall happens through `recall(...)`,
model-directed memory operations happen through ordinary tools, completed-turn
observation happens through `record(...)`, and write/consolidation happens
through `finish(...)`. Add finer hooks such as `after_tool_result` only after a
concrete experiment cannot be expressed with this shape.

## Context Injection

When memory becomes model-visible, inject it as ordinary messages. The default
shape should be a `RuntimeMessage` with `sender="memory"` and `kind="context"`:

```python
runtime_message(
    "Relevant memory:\n...",
    sender="memory",
    target=agent.name,
    kind="context",
)
```

That shape is already compatible with `State`, `build_context_view`, trace
printing, and context compression policy. LLM-backed agents fold runtime
system messages into the single provider-facing system prompt before the model
call, so memory stays trace-visible without creating multiple system messages
on the wire. Implementations may choose a user message only when the memory is
intentionally framed as user-provided context.

Prompt-only injection should be treated as a special case and should stay
visible in trace metadata or recorded messages. Hidden prompt mutation makes it
hard to explain why the model knew something.

## Tool Exposure

Memory tools should use the existing tool boundary:

- `search_memory`: model-directed recall.
- `write_memory`: explicit model-requested persistence.
- `edit_memory`: maintenance of a file-backed or structured memory.

The runtime should not need a special memory-tool path. A memory-backed tool is
an `AgentTool`; the provider bridge already knows how to describe tools to the
model, and `dispatch_tool_calls` already records tool execution events.

## Persistence Backends

Backends are implementation details. Possible backends include:

- no-op or in-memory stores for tests and demos;
- `FilesystemMemory`, which injects a filesystem memory directory policy, writes
  bounded host-owned evidence files at run end, keeps retained `INDEX.md`
  summary links valid, and can apply an optional distiller result to
  `summary.md`, `INDEX.md`, `MEMORY.md`, and `memory_summary.md`. The distiller
  returns the full updated `MEMORY.md` handbook (the model owns merging,
  rewriting, and dropping lessons); `finish(...)` writes that rewrite when it
  passes size, reference, and erasure guards, keeps the prior handbook on an
  empty or rejected rewrite, and refreshes `memory_summary.md` when the
  distiller did not provide one;
- SQLite or JSONL records;
- a vector index;
- a remote memory service.

`FilesystemMemory.finish(...)` serializes the complete
read-distill-commit operation with one inter-process lock per memory root. The
lock includes the optional model call because `MEMORY.md` is a full rewrite and
because the distiller may choose the final namespace only after reading prior
memory. Temporary-file replacement remains responsible for per-file atomicity;
it is not a substitute for this logical single-writer boundary. A container
that mounts only one namespace child must separately bind the shared host root
lock directory into `<memory-root>/.memory-lock/`; mounting one lock file is not
enough because `filelock` may unlink and recreate it. The host chain runner also
runs root-wide maintenance before and after a child-isolated batch, because one
child mount cannot count sibling namespaces. The Docker mount setup creates the
lock directory and file on the host first. A root-run worker uses that shared
owner to transfer new and legacy visible memory back to the host owner while
keeping private directory/file modes; it never solves cross-UID access by making
transcripts world-readable or writable. See ADR
[serialize-filesystem-memory-consolidation](../decisions/20260714-serialize-filesystem-memory-consolidation.md).
Storage, retry, and routing limits are recorded in ADR
[bound-filesystem-memory-growth](../decisions/20260714-bound-filesystem-memory-growth.md).

Keep the package facade small. Top-level imports should expose the memory
protocol and complete memory implementations. For the starter mechanisms, keep
implementation-specific helpers in the same file as their mechanism rather
than creating helper modules.

The abstraction should not require a backend when the method only injects
static context, nor should it require retrieval when the method only exposes a
filesystem directory and lets the model inspect files through tools.

## Trace And Events

Memory should be observable before it becomes clever. Future implementation
work should consider explicit memory events such as:

- `MemoryRecallStartEvent` / `MemoryRecallEndEvent`
- `MemoryInjectionEvent`
- `MemoryWriteEvent`

The first implementation may instead record injected messages and use existing
tool events, but it should not leave recall or writes invisible. If memory
retrieval becomes a durable operation in the runtime, add event pairs and derive
memory spans in `trace/spans.py`.

## Expected Fits

| Memory method | Memory pieces |
| --- | --- |
| Run summary memory | `finish(...)` learner plus `initial(...)` injection |
| User preference memory | explicit memory tool plus stable `initial(...)` context |
| File-system memory | backend directory plus injected instructions or filesystem tool exposure |
| Vector memory | `recall(...)` plus snippet injection |
| Self-editing memory | memory tools plus write policy and trace events |
| Read-only project memory | `MemorySource` plus `initial(...)` context, no sink |

## Imported Mechanism Fit

The migrated reproduction spec under `docs/reference-architectures/` (gitignored
local notes) includes SWE-bench-focused memory patterns. Treat its benchmark-specific wording as an
example domain, not as a Simple Agent Lab requirement. The filesystem-memory
shape is useful because it exercises prompt injection, ordinary file tools,
filesystem-backed storage, post-run learning, and memory-only model calls.

### Filesystem Memory

Fit: implemented as `FilesystemMemory`.

Generalized shape:

- The memory backend is a directory tree grouped by a project, user, benchmark
  suite, market, patient cohort, workshop, or other memory namespace.
- At run start, the model receives a short policy block with either the exact
  memory path or, when no namespace is known yet, the memory root and available
  namespace names.
- The model reads memory through existing file tools such as bash or an MCP
  filesystem server.
- At run end, the host can write bounded `task.md`, `transcript.md`, and generic
  `artifacts/*` snapshots, then optionally runs a no-tools distillation pass to
  update concise summaries, the run index, and a durable handbook. A successful
  distiller no-op (`retain_run=false`) skips the evidence write entirely.

Default layout:

```text
~/.simple/memory/
└── {memory_name}/
    ├── memory_summary.md
    ├── MEMORY.md
    ├── INDEX.md
    └── runs/
        └── {run_id}/
            ├── task.md
            ├── transcript.md
            ├── artifacts.md
            ├── artifacts/
            │   └── submission.txt
            ├── memory_error.md   # optional, only when memory writing fails
            ├── summary.md
            ├── .pending.json     # writing or bounded prepared-plan journal
            └── .commit.json      # host-owned idempotency/ordering metadata
```

Mapping to this sketch:

- The directory is a persistence backend, but not necessarily a
  `MemorySource`; recall is model-driven through normal file tools. Do not add a
  default `recall(...)` implementation that silently searches and injects
  filesystem snippets before every model request.
- The path/policy block is `initial(...)` context injection.
- The existing bash or MCP filesystem tool provides tool exposure; the memory
  layer may contribute no tools of its own.
- Evidence writing and summary/index updates are `finish(...)` learner/sink
  work. Bounded run evidence and a minimal `summary.md` are still written when
  optional distillation fails, so retained `INDEX.md` rows never point at a
  missing summary. Distillation failures can leave a compact `memory_error.md`
  marker while skipping only learned `MEMORY.md` updates. A successful
  `retain_run=false` result is the disk no-op path.
- `MEMORY.md` is a single model-owned handbook. The distiller is shown the
  current `MEMORY.md` and returns the complete updated file, doing its own
  merging, rewriting, deduplication, and pruning; `finish(...)` keeps no
  deterministic section/promotion logic. Determinism lives only in the
  guardrails: an empty rewrite keeps the prior handbook, and a rewrite that is
  oversized, structurally empty, or would erase every lesson is rejected (the
  prior handbook is kept and a `memory_error.md` marker records the skip).
  `memory_summary.md` is still rebuilt from `INDEX.md` when no model-provided
  summary was accepted.
- `make_filesystem_distiller(provider)` builds an explicit no-tools LLM
  distiller. It chooses `retain_run` and `memory_name` after seeing the completed
  run evidence, then returns a per-run summary, index row, the full rewritten
  `MEMORY.md` handbook (`memory_md`), and optionally a refreshed
  `memory_summary.md` for that namespace. Use the same
  provider as the main agent when you want the same model, but keep the extra
  model call visible in code rather than implicit. Put this in the high-level
  assembly code that already has the provider:

```python
memory = FilesystemMemory(distiller=make_filesystem_distiller(provider))
binding = memory.bind(
    MemoryContext(
        agent="agent",
        task=task,
        run_id=run_id,
        session_id=session_id,
    )
)
agent = make_llm_agent(
    name="agent",
    provider=provider,
    tools=[*base_tools, *binding.tools],
    hooks=binding.hooks,
)
state, events = agent.run(task)
```

The selected `memory_name` becomes `{memory_root}/{memory_name}`, where the
default `memory_root` is `~/.simple/memory`. Callers can pass `root=...` to
place filesystem memory under an eval output directory, mounted container
directory, workshop directory, or any other host-owned location:

```python
memory = FilesystemMemory(root=eval_run_dir / "memory")
```

If the agent Python process runs inside a local Docker eval container, keep this
as a memory-path choice and let evals handle the container mechanics:
`LocalDockerBackend(memory_home=host_memory_dir)` bind-mounts the directory
read-write and exposes `SAL_MEMORY_HOME` inside the container. Memory
implementations should not import Docker or evals; they should only receive the
local path chosen by the assembly code, such as `FilesystemMemory(root=...)`.

The run directory is `runs/{run_id}`; if `run_id` is omitted, the implementation
falls back to `session_id`, then a microsecond timestamp. The bounded evidence
fingerprint makes an exact retry idempotent, before another distiller call. If a
caller intentionally reuses an id with different evidence, the second logical
run receives a deterministic `--{content_hash}` suffix; it never globally
rewrites old handbook references.

`FilesystemMemoryLimits` holds the public budgets. Defaults keep 64 runs and at
most 128 MiB per namespace, plus root-wide targets of 1,024 runs, 512 MiB, and
128 namespace directories. The SWE memory-chain runner may raise only the last
value for a fresh run-local root to the already-known finite planned namespace
count and records it in `experiment.json`; a reused `--memory-home` stays at 128
unless the operator explicitly passes a positive
`--memory-max-namespaces`. Task/transcript limits are 20,000/1,000,000 bytes;
one run keeps at most 16 artifacts, 500,000 bytes each and 1,000,000 bytes in
total. Run and navigation summaries are capped at 12,000 characters, and each
index cell at 2,000. The handbook cap is 20,000 characters, with at most 32 run
references per namespace and a 128-reference root-wide target (a hard rewrite
guard only when the writer can see the full root). The final
distiller prompt is capped at 512 KiB of UTF-8 bytes; the built-in provider
distiller lowers that limit when provider context-window metadata requires it.
The minimum supported rendered-prompt ceiling is 16,000 bytes; constructing the
built-in distiller fails clearly if its output and safety reserves leave less
usable input room. Text writers pin LF newlines on every platform, so Windows
newline translation cannot make persisted evidence exceed the UTF-8 byte count
used by these budgets.

Retention runs under the same consolidation lock before and after commit. It
keeps the current run and evidence cited by `MEMORY.md`, removes the oldest
other runs until namespace and root count/byte targets hold, filters `INDEX.md`,
and rebuilds `memory_summary.md`. Every deletion is verified before counts or
bytes are reduced. A permission failure therefore leaves the run visible and
produces a stable warning instead of false convergence. A post-commit deletion
failure can temporarily leave the count at its target plus one; all subsequent
evidence for that namespace or root is refused until maintenance actually
removes the excess. The current maintenance result enforces this even if the
warning path is unwritable. Early `writing` pending
directories are removed across all namespaces. A v2 `prepared` marker contains
a bounded commit plan and before/proposed handbook hashes; the next lock holder
forward-completes it without another model call or a duplicate handbook lesson.
If that recovery remains blocked, both retention passes pin the pending run and
the namespace refuses new evidence until recovery succeeds; a persistent file
error therefore overwrites stable warnings instead of adding one prepared run
per attempt.
A failed atomic replacement normally deletes its hidden `.*.tmp` file. If the
filesystem also refuses that deletion, memory creates one direct, stable
`.memory-write-blocked` sentinel in the affected namespace (or shared
`.memory-lock/root-write-blocked` for a root-level write) and stops all further
writes visible in that scope. A later
maintenance pass removes the sentinel only after it can inspect the same scope
and delete all stale temps. Namespace-local placement prevents a child-only
container from clearing a sibling namespace's block merely because that sibling
is outside its mount view; `SAL_MEMORY_ROOT_VIEW=isolated` also makes every
child honor, but never clear, the shared root-level block.
A legacy handbook that pins too much evidence, root-wide citation pressure, or
an unremovable tree produces one stable `retention_warning.md`; it does not
create a new error file per run.

If the caller passes `memory_name` explicitly, `initial(...)` can expose that
memory directory before the run so the model may inspect prior memory. If the
caller omits it but prior memory namespaces exist, `initial(...)` can expose the
root and at most 64 recent namespace names for lightweight selection; each
description is at most 200 characters and the names section is at most 8,000
bytes. The
distiller still chooses the final namespace after the completed transcript and
artifacts are available, but receives detailed context for at most eight recent
namespaces. Automatic routing may create up to 64 learned namespaces and then
uses stable `default`; a missing/failed distiller also uses `default`, never the
session id. A new explicit caller namespace is never redirected, but a general
or reused root admits no more than 128 by default. Batch admission checks the
whole requested set under one lock and fails closed when adding it would cross
the configured finite cap. If a new layout fails, the other newly-created batch
layouts are rolled back without touching existing namespaces. Admission never deletes an
empty-looking namespace because that directory may belong to an active run
between `initial(...)` and `finish(...)`. An already-existing namespace remains
usable when a legacy root is over the cap, which allows consolidation without
permitting further directory growth.

Namespace components that are already portable ASCII names stay unchanged.
Names containing separators or surrounding whitespace, names longer than 80
characters, root-reserved filenames, Windows device basenames, and names using
the internal `salm-` prefix are encoded with a 20-hex-character SHA-256 suffix.
This avoids the old lossy `a/b` -> `a_b` collision. Exact physical names returned
by `available_memories()` round-trip as selectors, while a new raw name cannot
impersonate a generated path; aliases that differ only by case are refused so a
root created on Linux remains unambiguous on case-insensitive filesystems. The
SWE chain runner uses a separate `salx-` encoding whose output is already a
filesystem-memory fixed point. Ambiguous directories made by the old lossy
encoder are not guessed or silently reused and should be migrated explicitly.
Only real namespace directories are eligible for recall or writes. A namespace
symlink is still counted conservatively for admission and retention, but
maintenance fails closed instead of traversing, changing ownership of, or
creating skeleton files in its external target.

Root-wide run/byte/reference values are retention targets, while namespace
admission and per-run evidence budgets provide the deterministic new-growth
boundary. Per-namespace run/byte/reference values preserve cited or unreadable
legacy overages with a warning; a handbook already over its reference target is
accepted only when the rewrite strictly decreases that count.
In a child-isolated eval container, sibling handbooks are invisible, so the
128-reference root target cannot be a hard write gate: host maintenance detects
and warns on an overage before/after the batch. New nondecreasing rewrites are
guarded at 32 references per namespace; together with the configured finite
namespace cap (128 by default, or the recorded planned count for a fresh
run-local eval), new cited evidence cannot grow without bound. Reaching a root
target never
authorizes deleting cited evidence or a whole namespace automatically.

A complete `MEMORY.md` replacement is accepted only if that target handbook was
loaded in full. For one-pass dynamic routing, a run may be indexed into an older
namespace whose handbook was not among the detailed contexts; in that case the
host keeps the existing handbook and writes a compact rejection marker instead
of risking silent lesson loss.

Domain-specific details to generalize:

- Use an LLM-selected memory namespace, with an explicit caller-provided
  `memory_name` only as an override.
- Put domain outputs under `artifacts/` instead of hard-coding `patch.diff`.
  SWE can store a patch as `submission.txt`; finance can store a portfolio JSON;
  medical tasks can store extracted findings or recommendations.
- Keep index fields generic: `Summary`, `Scope`, `Signals`, `Keywords`, and
  `Artifacts` instead of SWE-only files, symbols, tests, and errors.
- Preserve the broader rule: bounded evidence snapshots are host-written,
  distilled memory is concise, and official benchmark labels or scoring
  outcomes must not leak into future runs. Complete audit trajectories belong
  in the trace or artifact store, not the active memory cache.
- Borrow the progressive disclosure shape from Codex without the full pipeline:
  `memory_summary.md` is the cold-start navigation file, `MEMORY.md` is the
  durable handbook, `INDEX.md` routes to per-run evidence, and transcripts stay
  behind targeted search. Borrow the Claude/Claw-style instruction loading
  principle of small prompt budgets and explicit file paths, but do not add
  project-rule discovery, RAG, daemons, or dreams to this starter mechanism.
- Keep filesystem recall Codex-style: prompts tell the model when and how to do a
  quick memory pass; the runtime does not prefetch or inject matching lines via
  a pre-model-request hook point. Use `recall(...)` for memory methods whose
  core shape is runtime retrieval, such as vector snippets.

Abstraction notes:

- The memory layer must support memory methods that expose a directory and
  instructions rather than a direct recall API.
- It must support a no-tools model call for distillation so memory updates do
  not depend on or pollute the main agent tool surface.
- It needs run metadata such as `session_id`, `run_id`, an optional explicit
  memory namespace key, and optional `step_index`; these should stay outside
  `Message` and flow through memory hook context.

### Refinements From The Imported Spec

The imported mechanisms suggest the high-level memory layer should include these
pieces before adding a concrete backend:

- A hook context carrying `agent`, `state`, `session_id`, an optional memory
  namespace key, and domain metadata.
- A prompt-block injection path in addition to message injection, with trace
  visibility for whichever path is used.
- Additive memory tool registration before the first model request.
- A transcript text extraction boundary that explicitly excludes raw provider
  dumps and off-band debug data.
- A memory-only model call helper for summarization or consolidation with no
  ordinary tools attached.
- A best-effort shutdown rule: memory indexing or consolidation failures should
  not fail an otherwise valid agent run.

## Implementation Sequence

Completed starting point:

1. Protocol/data types and a no-op memory implementation.
2. MCP-like memory binding that produces `tools` plus future lifecycle hooks,
   without changing the current agent runtime.
3. Focused tests for binding shape, transcript extraction, and filesystem
   evidence/distillation writes.
4. Concrete `FilesystemMemory` implementation.

Possible next steps:

1. Add a deterministic run script once there is a stable public demo story.
2. Add memory events only if message injection and ordinary tool events are not
   enough for trace inspection.
3. Add vector or remote backends as separate implementations, not as core
   requirements.

## Open Questions

- Should future memory injection happen before every model request, or is
  run-start injection enough for the first evaluation wave?
- Are message injection and ordinary tool events sufficient trace visibility, or
  should memory-specific events be added?
- Should sub-agents inherit parent memory by default, or require explicit
  memory policy?
- Should write-capable memory tools require human approval in demos?
