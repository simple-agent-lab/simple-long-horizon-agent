# Reference Architecture: Context Management Pipelines

## Source

- Local source: `/Users/bytedance/Downloads/package 3/src/query.ts`
- Local source: `/Users/bytedance/Downloads/package 3/src/services/compact/autoCompact.ts`
- Local source: `/Users/bytedance/Downloads/package 3/src/utils/toolResultStorage.ts`
- OpenAI Codex source: `https://github.com/openai/codex`, reviewed at commit `f7e8ff8`
- OpenAI Codex files: `codex-rs/core/src/context_manager/history.rs`,
  `codex-rs/core/src/context_manager/normalize.rs`,
  `codex-rs/core/src/session/turn.rs`, `codex-rs/core/src/compact.rs`
- Date reviewed: 2026-05-07

## Summary

Both references treat context as a projection over durable history, not as the
history itself.

The local `package 3` source uses a layered query-time pipeline:

```text
full messages
  -> tool result budget
  -> optional history snip
  -> microcompact
  -> context collapse projection
  -> autocompact
  -> model request
```

OpenAI Codex uses a `ContextManager` around raw response items. It records the
full history, normalizes model-facing prompt items, estimates token usage, keeps
call/output pairs valid, and replaces history with compacted summaries when a
turn crosses the model's auto-compact limit.

## What To Borrow

- Keep full history durable and inspectable.
- Make the model-visible view an explicit projection step.
- Normalize tool call and tool result pairs before model calls.
- Prefer cheap, low-loss transforms before lossy summarization.
- Track budget statistics in the same place that builds the view.
- Keep compaction as a future layer over the projection, not the first API.

## What To Avoid For Simple Agent Lab

- Do not copy a production-sized `ContextManager` into the teaching runtime.
- Do not introduce provider-specific token accounting in the core.
- Do not make context collapse, remote compaction, cache editing, or session
  memory part of the first `context_view` implementation.
- Do not hide visibility decisions inside model adapters.

## Simple Agent Lab Shape

The useful teaching boundary is:

```text
State.events -> state.messages -> build_context_view(...) -> model_messages(...)
```

`context_view` should answer:

- Which messages are visible to this agent?
- Which messages were dropped by policy or budget?
- What rough size did the projected view have?
- Were large message bodies clipped?
- Did tool-call/tool-result pairs remain valid?

The current implementation keeps the old beginner API:

```text
context_view(agent, state) -> list[Message]
```

and adds a detailed projection for experiments:

```text
build_agent_context_view(agent, state, policy=ContextPolicy(...)) -> ContextView
```

This is enough for classroom experiments with visibility, recency, and budget
pressure without committing the repo to a full production compaction system.
