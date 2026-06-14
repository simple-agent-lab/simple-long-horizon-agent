---
title: "Recoverable Compression and Agent-Controlled Compaction"
status: Accepted
date: 2026-06-12
slug: recoverable-compression-and-agent-compaction
---

# Recoverable Compression and Agent-Controlled Compaction

## Context

The compression layer shipped two system-controlled strategies
(`ToolCompactStrategy`, `SummarizeStrategy`): the runtime watches a token
threshold and folds older messages into a summary. Two gaps stood out when
mapping the layer against the agent context-compression literature (the
survey taxonomy behind YerbaPage/Awesome-Context-Compression):

1. **Compression was lossy from the model's point of view.** The transcript
   is append-only — folded messages never leave `state.messages` — but
   nothing told the model where they went or gave it a way to read them
   back. The dominant failure mode of summarization (a detail dropped by the
   summary is needed three steps later) had no recovery path.
2. **Only the system decided when to compress.** The control-policy spectrum
   runs from threshold triggers to the agent managing its own context
   (Context-as-a-Tool / AgentFold-style); a lab whose point is comparing
   agent designs had only one end of it.

## Decision

Two capabilities, designed as halves of one loop:

- **Summaries cite, recall retrieves.** Every compression replacement ends
  with a `source_note(...)` footer naming the transcript indices it folded
  ("[Compressed from transcript messages 2-8. ...]"). The new
  `tools.recall.make_recall_tool(state)` is the retrieval side: the model
  passes cited indices and gets the original messages rendered verbatim
  (read-only, capped per message). Compression becomes recoverable
  externalization instead of deletion.
- **Agent-controlled compaction splits decision from application, routed
  through the transcript.** `compression.agent_control.make_compact_control()`
  returns a paired `compact` tool and strategy. The model calls
  `compact(summary=...)` with its own replacement text; the tool stays a pure
  function and returns the request as structured `ToolResult.details`
  (`{"compact_request": ...}`). The dispatch loop records that into the
  tool_result bundle's `details` sidecar — single-threaded, at the one safe
  point — so the request enters the append-only event log (and the trace) like
  any other tool output, with no shared mutable state between the two halves.
  The paired strategy runs at the next turn start — the loop's one safe
  compression point, after the tool_result bundle is recorded, so a fold can
  never split an in-flight tool_call from its result — reads the most recent
  `compact_request` back off the active view, and applies it. Exactly-once
  needs no destructive read: the strategy applies a request only while it is the
  newest message in the view, and applying splices a `kind="summary"` that
  outranks it, so it is never the max again. The existing
  `maybe_compress_context` runtime (pair alignment, sizing,
  `ContextCompressionEvent`) is reused unchanged.

## Consequences

- Aggressive compaction gets cheaper to risk: anything a summary drops can
  be recalled by index, addressing the post-compression-access failure mode.
- The control-policy spectrum is now demonstrable side by side: threshold
  strategies, agent-controlled compaction, or both via `TieredStrategy`
  (agent stage first).
- `make_recall_tool` closes over a `State`, so it pairs with the
  module-level `run(agent, state)` composition rather than `Agent.run(task)`.
- The `compact` tool and its strategy share no mutable state, so one
  `CompactControl` can serve any number of concurrent runs; the request lives
  in the event log, so it is visible in the trace and recoverable like any
  other tool output.
- A request that finds nothing old enough to fold is dropped, not deferred:
  because it only applies while it is the newest message, it can never fold a
  later turn's content under a stale summary — no surprise compactions turns
  later.
- Representation-level compression (KV cache, visual tokens) and learned
  compression policies stay out of scope: the former needs model internals
  an API client cannot reach, the latter is heavier than the teaching path
  warrants.

## Alternatives Considered

- **Apply compaction inside the tool call.** More direct, but the tool runs
  in the dispatch worker pool where recording events is not thread-safe, and
  a mid-batch fold could orphan the in-flight tool_call whose result is not
  yet recorded. The deferred design reuses the existing safe point instead.
- **A shared in-memory request holder between tool and strategy.** The first
  cut: the tool wrote the request to a single-slot mailbox the strategy read.
  It worked but stood outside the event-sourced design — the request never
  entered the log or trace, the holder was mutable state shared across the
  parallel tool pool (needing its own locking), and one `CompactControl` could
  serve only one run. Routing the request through the tool_result sidecar
  instead reuses the loop's existing safe recording point, makes the request a
  first-class log entry, and drops the shared state entirely.
- **Mention the recall tool only when it is wired.** Summaries would need to
  know the agent's toolset; instead the footer is unconditional provenance
  ("when it is available"), which stays truthful either way.
