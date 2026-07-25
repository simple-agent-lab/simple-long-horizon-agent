# Docker Live Trace Contract

Any agent running inside Docker can expose the same incremental
`trajectory.jsonl` the host trace viewer polls. The API is in
`simple_agent_lab.trace`.

## Mount contract

Prepare a run directory on the host, bind it into the container at a stable
path, and point the trace file inside the mount so writes are visible on the
host:

```bash
docker run -v "$HOST_RUN:/agent/run:rw" ...
--traces /agent/run/out/trajectory.jsonl   # or export LIVE_TRACE_PATH=...
```

SWE-bench uses `/agent/run` by default. `LIVE_TRACE_PATH` is
`simple_agent_lab.trace.LIVE_TRACE_PATH_ENV`; the host layout is
`HOST_RUN/{input,out}/`, with the trace at `out/trajectory.jsonl`.

## In-container usage

Wrap event consumption in `LiveTraceSession`, then write the canonical final
record after any post-run enrichment (patches, labels):

```python
from simple_agent_lab.trace import LiveTraceSession, TraceMeta, write_canonical_trace

state, events = agent.run(task, max_turns=75)
meta = TraceMeta(trace_id="my.run.001", producer="suite:example")
with LiveTraceSession("/agent/run/out/trajectory.jsonl", state,
                      trace_id=meta.trace_id, producer=meta.producer):
    list(events)
# ... post-run state updates ...
write_canonical_trace("/agent/run/out/trajectory.jsonl", state=state, trace_meta=meta)
```

When nothing runs after the event loop, `run_agent_with_live_trace(agent, task,
path, trace_meta=..., max_turns=...)` does both steps.

## Viewing

```bash
bash runs/demos/run_trace_viewer.sh --dir <host-run>/out
```

The viewer tails **local** files. For a remote daemon, use host-pull or point it
where artifacts land. `scripts/run_live_trace_demo.py` runs the whole path
without Docker.

Related code: `src/simple_agent_lab/trace/live.py` (session + incremental
writer), `src/simple_agent_lab/trace/jsonl.py` (atomic JSONL IO), and
`src/simple_agent_lab/evals/in_container.py` (generic runner, trace defaults).
