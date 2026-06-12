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
- **Agent-controlled compaction splits decision from application.**
  `compression.agent_control.make_compact_control()` returns a paired
  `compact` tool and strategy sharing a single-slot request holder. The
  model calls `compact(summary=...)` with its own replacement text; the tool
  only records the request (tool executes run in the worker pool, where
  mutating `State` is out of bounds). The paired strategy applies it at the
  next turn start — the loop's one safe compression point, after the pending
  tool_result bundle is recorded, so a fold can never split an in-flight
  tool_call from its result. The existing `maybe_compress_context` runtime
  (pair alignment, sizing, `ContextCompressionEvent`) is reused unchanged.

## Consequences

- Aggressive compaction gets cheaper to risk: anything a summary drops can
  be recalled by index, addressing the post-compression-access failure mode.
- The control-policy spectrum is now demonstrable side by side: threshold
  strategies, agent-controlled compaction, or both via `TieredStrategy`
  (agent stage first).
- `make_recall_tool` closes over a `State`, so it pairs with the
  module-level `run(agent, state)` composition rather than `Agent.run(task)`.
- One `CompactControl` serves one running agent; concurrent runs need their
  own (the request holder is shared mutable state between its two halves).
- A request that finds nothing old enough to fold is consumed silently, not
  deferred — no surprise compactions turns later.
- Representation-level compression (KV cache, visual tokens) and learned
  compression policies stay out of scope: the former needs model internals
  an API client cannot reach, the latter is heavier than the teaching path
  warrants.

## Alternatives Considered

- **Apply compaction inside the tool call.** More direct, but the tool runs
  in the dispatch worker pool where recording events is not thread-safe, and
  a mid-batch fold could orphan the in-flight tool_call whose result is not
  yet recorded. The deferred design reuses the existing safe point instead.
- **Signal the request through `state.data` or message sidecars.** Both
  bind the strategy to a specific `State` or overload message metadata; the
  private holder keeps the pairing explicit and lets the control work with
  `Agent.run(task)`-created states too (for the compact half).
- **Mention the recall tool only when it is wired.** Summaries would need to
  know the agent's toolset; instead the footer is unconditional provenance
  ("when it is available"), which stays truthful either way.
