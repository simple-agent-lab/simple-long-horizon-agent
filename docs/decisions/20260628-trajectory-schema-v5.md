---
title: "Trajectory Schema v5 — Append-only Event Stream"
status: Accepted
date: 2026-06-28
slug: trajectory-schema-v5
---

# Trajectory Schema v5 — Append-only Event Stream

## Status

Accepted

## Context

The previous (intermediate) schema made the record canonical-vs-external and
lifted the system prompt into an `agents` registry. But the file was still **one
JSON object per run**: `{schema, …, agents, events:[…], messages:[…], spans:[…],
model_turns:[…], cost, meta}`, rewritten in full on every live flush (atomic
tmp+rename). For a long run that means re-serializing and re-writing a growing
multi-MB blob every ~2s — O(run length) work per flush, and a 200 MB single
line at the end.

Two facts make that wasteful:

1. **`events` is the whole trace.** `messages` is exactly the payloads of the
   `message` events; `spans`, `model_turns`, and `cost` are pure functions of
   `events`. The viewer already proves this — it ignores the embedded arrays and
   recomputes them in JS (`spansFromEvents`, `modelTurnsFromEvents`). So the
   embedded derived layers are dead weight the reader never reads.
2. **Events are append-only by nature.** The runtime emits them one at a time
   and never edits a past one. Rewriting the whole file to add one event fights
   that grain.

## Decision

Persist the trace as an **append-only JSONL event stream**: a header line plus
one line per event. Bump the schema to `simple-agent-lab.trajectory.v5`.

```jsonc
// line 0 — header, written once
{"schema":"simple-agent-lab.trajectory.v5","type":"trajectory","trace_id":"…",
 "producer":"…","task":"…","meta":{…}}
// line 1..N — one event per line, APPENDED as they happen
{"index":0,"elapsed":0.0,"kind":"message","message":{…}}
{"index":1,"elapsed":0.1,"kind":"agent_start","agent":"…","system_prompt":"…"}
{"index":2,"elapsed":0.2,"kind":"model_request","agent":"…", … }
…
```

### Live writes append; they never rewrite

The incremental writer tracks a high-water mark (events + raw blobs already
written) and on each flush **appends** only the new event lines (and new raw
blobs to the sibling pool). The header is written once. Crash/torn-read safety
becomes the log-tailing contract: every line is one complete JSON object, so a
reader skips an incomplete trailing line — no tmp+rename of the whole file.

The container eval writer (`in_container.py`) pushes through an artifact store
whose `put` replaces the whole object, so it writes the full stream (header +
all events) each flush — same *format*, different *write strategy*. Append is
the local-file (`live.py`) optimization the store path can't use.

### Everything else is derived by the reader

`messages`, `spans`, `model_turns`, `cost`, and the `agents` registry are **not**
persisted. Readers derive them from the event stream:
- python already has `messages` (the `message` events), `spans_from_events`,
  `model_turns_from_events`, `RunCost.from_run`;
- the viewer already derives spans/turns; it gains a tiny `collectAgents` over
  the event stream, mirroring the python `collect_agents`.

### The system prompt rides on events

The system prompt is the one request field not otherwise in the stream. It is
carried on the events that introduce an agent:
- `agent_start` gains `agent` + `system_prompt` (every `run()` — main and each
  sub-agent emits one);
- the context **compressor** does not go through `run()` (it is a single
  `generate`), so its `model_request` carries `system_prompt`.

`collect_agents` walks the stream and registers `agent -> system_prompt` from
whichever event carries it (first wins). Dynamically-spawned sub-agents register
when their `agent_start` appears — which a once-written header could not capture.

### Each event carries a stable `uuid`

`State.record_event` stamps every event with a UUID4 (`_BaseEvent.uuid`),
independent of its positional `index`. In an append-only log that survives
merges, resumes, and cross-file references, a positional index is not a stable
identity; the `uuid` is (cf. Claude Code's transcript `uuid`). The `index` stays
as the chronological ordinal.

### The header `task` is a one-line preview

The full task already rides in the first event (its `task` message), so the
header stores only a short title (`task_preview` — first meaningful line,
clamped, skipping markdown headings and the SWE-bench preamble), mirrored by the
viewer's `taskPreviewLine`. The header never re-stores a multi-KB task.

### `llm_payload` is dropped from serialized events

As before, the reconstructable per-turn `llm_payload` is not persisted (the
reader rebuilds the request from `messages` + `agents`, or reads the verbatim
wire from the external pool). The `slim`/drop step now runs per event line.

## Schema shape (v5)

```text
trajectory.jsonl       — line 0: header; lines 1..N: one event each (append-only)
trajectory.jsonl.raw.jsonl — one compact provider raw blob per line; raw_ref = line index
```

Removed vs the previous schema: the embedded `messages` / `spans` /
`model_turns` / `cost` arrays and the `agents` map (all reader-derived now).
Added: `system_prompt` on `agent_start` and (compressor) `model_request`; the
file is a stream, not one object.

## Migration

- **Viewer / readers are v5-only.** The viewer loader reads the header line
  (schema, no `events`) followed by event lines and builds the in-memory
  `trace` from the stream. Backward-compatibility with the prior single-object
  schema was intentionally dropped — it was an intermediate state.
- **No on-disk migration**: nothing scores off old traces.
- **Producers**: `run_trace.py` (header + event-line serializers), `live.py`
  (append writer), `in_container.py` (full-stream-per-flush), `core.py`
  (`agent_start` fields), `compression/strategies.py` (compressor
  `system_prompt`).
- **Golden fixture + tests** regenerate to the stream shape.

## Consequences

**Positive**
- Live flush is O(new events), not O(run length); no multi-MB rewrite loop.
- The file is a true append-only log — simple to tail, reason about, and recover.
- Smaller files: the redundant derived arrays are gone.
- One source of truth (events); no chance of embedded-vs-derived drift.

**Negative / cost**
- Another format bump close on the heels of the prior schema, and a coordinated
  producer + viewer + fixture change.
- Readers must derive (the viewer already did; python already has the helpers),
  so a brand-new reader has slightly more to do than reading an embedded array.

## Alternatives considered

- **Keep one record, just append a delta record per flush.** Re-introduces
  embedded-vs-derived ambiguity and a merge step; the event stream already *is*
  the deltas.
- **Header carries `agents`.** A once-written header can't grow when a sub-agent
  appears mid-run; events can. Rejected in favour of deriving from the stream.
- **Stop deriving in the viewer; embed spans/turns/cost.** That is exactly the
  dead weight this ADR removes; the viewer already recomputes them.
