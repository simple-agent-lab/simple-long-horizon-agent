# Docker live trace contract

Any agent that runs inside Docker can expose the same incremental
``trajectory.jsonl`` the host trace viewer already polls. The reusable API lives
in ``simple_agent_lab.trace``.

## Host layout

Before ``docker run``, prepare a run directory on the host:

```text
HOST_RUN/
  input/          # optional inputs (instance JSON, etc.)
  out/
    trajectory.jsonl   # live: append-only event stream; final: one canonical record
```

While a run is in flight, ``trajectory.jsonl`` is an **append-only event
stream** — a header line (``{"type":"trace_header",...}``) followed by one
``event_record`` per line. Each flush appends only the events added since the
last one, so a long run costs O(new events) per flush instead of rewriting the
whole record every interval. On ``stop(final_flush=True)`` (or an explicit
``write_canonical_trace``) the file is materialized into the **single canonical
record** the non-live writer produces, so a finished file is unchanged for every
reader. ``trace_record_from_jsonl`` folds either shape back into one record; the
host trace viewer calls it transparently, so it can tail a live stream or open a
finished file with no difference. A crash mid-append can only lose the torn last
line (JSONL readers skip it), instead of the old whole-file rewrite.

## Mount

Bind the host run root into the container at a stable path:

```bash
docker run -v "$HOST_RUN:/agent/run:rw" ...
```

SWE-bench uses ``/agent/run`` by default (see ``evals/swebench/suite.py`` and the
SWE-bench container half in ``simple_agent_lab.evals.suites.swebench``).

## Container path

Pick a trace file under the mount so writes are visible on the host:

```bash
# CLI flag (SWE-bench runner)
--traces /agent/run/out/trajectory.jsonl

# Or standard env var (optional; runners may read it when --traces is omitted)
export LIVE_TRACE_PATH=/agent/run/out/trajectory.jsonl
```

``LIVE_TRACE_PATH`` is defined as ``simple_agent_lab.trace.LIVE_TRACE_PATH_ENV``.

## In-container runner

1. Build ``State`` (or call ``agent.run`` to obtain ``state`` + ``events``).
2. Wrap event consumption with :class:`~simple_agent_lab.trace.LiveTraceSession`
   or call :func:`~simple_agent_lab.trace.run_agent_with_live_trace`.
3. After post-run enrichment (patches, labels), write the canonical final record
   with :func:`~simple_agent_lab.trace.write_canonical_trace` or let the
   helper do it when no post-run work is needed.

Minimal pattern when the agent loop is already set up:

```python
from simple_agent_lab.trace import LiveTraceSession, TraceMeta, write_canonical_trace

state, events = agent.run(task, max_turns=75)
meta = TraceMeta(trace_id="my.run.001", producer="suite:example", meta_fn=lambda: {"in_progress": True})
with LiveTraceSession("/agent/run/out/trajectory.jsonl", state, trace_id=meta.trace_id, producer=meta.producer, meta_fn=meta.meta_fn):
    list(events)
# ... post-run state updates ...
write_canonical_trace("/agent/run/out/trajectory.jsonl", state=state, trace_meta=meta)
```

One-liner when nothing runs after the event loop:

```python
from simple_agent_lab.trace import TraceMeta, run_agent_with_live_trace

state, events = run_agent_with_live_trace(
    agent,
    task,
    "/agent/run/out/trajectory.jsonl",
    trace_meta=TraceMeta("my.run.001", "suite:example"),
    max_turns=75,
)
```

## Host viewer

From the repo root, serve traces under the host output directory:

```bash
bash runs/run_trace_viewer.sh --dir evals/out/swebench_container_runs/<run_id>/<instance_id>/out
```

Or open a single file URL (see ``scripts/run_live_trace_demo.py``).

## Related code

- ``src/simple_agent_lab/trace/live.py`` — ``LiveTraceSession``, ``IncrementalTraceWriter``, and final-record helpers
- ``src/simple_agent_lab/trace/jsonl.py`` — atomic JSONL read/write
- ``src/simple_agent_lab/evals/suites/swebench/`` — SWE-bench container half (trace wiring)
- ``src/simple_agent_lab/evals/in_container.py`` — generic in-container runner and trace defaults
- ``scripts/run_live_trace_demo.py`` — local demo without Docker
