# Observatory — trace viewer

A single-file HTML viewer for `simple-agent-lab.trajectory.v3` trace records.

It is designed to make multi-layer agent traces (parent + sub-agents),
context compressions, tool errors, and the raw event stream all
visually scannable on one page. With the optional server it also doubles
as a live eval-companion: it watches `evals/out/` and surfaces new
trajectories as they finish, so you can read a run mid-eval.

```
studio/trace-viewer/
  index.html         self-contained viewer (HTML + CSS + JS, no build)
  serve.py           stdlib HTTP server: scans evals/out/, polling API
  sample-trace.jsonl hand-crafted demo trace with sub-agents,
                     a tool error, and a context compression
  README.md          this file
```

## Quick start (offline mode)

```bash
open studio/trace-viewer/index.html
```

The embedded sample loads on first paint, so the page is useful immediately.
Drag any `.jsonl` / `.json` trajectory onto the page (or click **file**) to
load it. This mode needs no server.

## Live eval mode (recommended for running evals)

```bash
bash runs/run_trace_viewer.sh
# → http://127.0.0.1:8765
```

This boots a tiny stdlib-only HTTP server that recursively scans
`evals/out/` (or a custom dir, see flags below), classifies every
`.jsonl` / `.json` file it finds, and exposes a polling API the viewer
hits every 5 seconds. A new **TRACES** button appears in the top bar
with a count of viewable trajectories; clicking it opens a picker
panel:

- **trajectories / all** toggle at the top — the default hides
  predictions / eval_result / instance_metadata files so only viewable
  records show.
- search box (substring across name, path, kind, trace_id, instance_id).
- grouped by run directory, sorted by modification time (newest first).
- each entry shows the kind badge, smart label (instance ID for
  SWE-bench layouts), age, and size.
- **follow latest** checkbox at the bottom — when on, the viewer
  auto-switches to the newest trajectory as soon as it appears on disk.
  Perfect for watching evals as they roll in.
- a green **NEW** badge marks every trace that appeared since the
  picker was last opened; the TRACES button also gets a `+N` pulse.

Server flags:

```bash
bash runs/run_trace_viewer.sh --port 9000
bash runs/run_trace_viewer.sh --dir evals/out/swebench_container_runs
bash runs/run_trace_viewer.sh --host 0.0.0.0      # expose on the network
```

The server walks recursively and detects trajectories by JSON shape
(`schema` field starts with `simple-agent-lab.trajectory`, or the
record contains an `events` array). For very large traces whose first
line exceeds the 96KB peek window it falls back to text-level
fingerprinting so multi-MB trajectories still get classified
correctly.

### Loading arbitrary files (without the server)

Drop a `.jsonl` or `.json` file onto the page or click **file**. The
loader accepts:

- a JSONL file with one trajectory record per line (the viewer loads
  the first record it can parse);
- a single-object JSON file containing a trajectory record.

Each record only needs `events`; `spans` and `model_turns` are
re-derived in the browser using the same logic as
`src/simple_agent_lab/trace/spans.py` and
`src/simple_agent_lab/trace/training.py`.

## Layout

Three panes plus a top stat strip:

- **Left — Structure (spans).** Hierarchical tree of `agent_run` →
  `turn` → `model_call` / `tool_call` / `compression`. Sub-agent
  spans are inlined under the `task` tool call that spawned them and
  shown in magenta. Each node carries its duration; tool errors get a
  red diamond, compressions a gold diamond.
- **Center — Timeline + Stream.** A waterfall SVG shows every span on
  a packed track with axis ticks. Below it, a filterable stream
  switches between three view modes:
  - `messages` — what was said (and thought), one row per
    transcript message, with sub-agent messages inlined chronologically
    and visibly indented.
  - `events` — raw event log including model requests/responses,
    tool start/end, turn boundaries, compression events.
  - `model turns` — one row per `model_request` → assistant response
    pair, with latency and token counts. Useful for fine-tune data review.
- **Right — Inspector.** Whatever you clicked (span, event, message,
  model turn) gets a full breakdown: typed metadata, content blocks
  rendered with role-specific styling, and a folded raw JSON dump
  where useful.   For model calls and model-request selections, **Wire debug ↗**
  opens a wire-debug panel with three tabs: **Raw request** (adapter
  `data.raw.request` when present, else reconstructed body), **Raw response**
  (`data.raw.response`), and **Export** (JSON / cURL / Python snippets).

The top **stat strip** surfaces the things that usually point to a
problem: error count (lit red), context compressions (amber),
sub-agent count (amber), total duration, token totals, and exit
reason. If anything is broken, you see it before you read.

## Why it looks like this

The design brief was "scientific instrument, not generic dashboard":

- IBM Plex Sans for prose and inspector titles; JetBrains Mono for
  labels, stats, timestamps, ids, and payloads.
- Cream paper on deep ink, with sharp accents — alarm orange-red for
  errors, mint phosphor for tool success, solar amber for context
  events, lapis for model activity, magenta for sub-agent nesting.
- Frame-number index column (`000`, `001`, …) like film reels;
  dotted grid in the waterfall; subtle SVG grain across the whole
  background.

The point is that a trace viewer is a diagnostic instrument: it
should reward dense reading and make anomalies pop, not look like
every other dashboard.

## Compatibility

The viewer reads `simple-agent-lab.trajectory.v3` records produced by
`simple_agent_lab.trace.trace_record(...)`. It re-derives spans
and model turns from `events` in the browser, so older records
without `spans`/`model_turns` work fine.
