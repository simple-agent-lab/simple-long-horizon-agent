# Evals

This directory holds higher-level behavior checks and comparison cases.

Unit tests live in `tests/`. Evals are for feedback that compares agent
behavior across prompts, recipes, context views, runtime designs, or model
adapters.

Trajectory collection for shared demos can live outside this directory because
trajectories are fact records, not scores. Scene-level suite adapters can keep
their collector beside their scorer when that makes the suite easier to read.
The first suite example is the local SWE-bench adapter smoke check:

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_containerized_agent \
  tests.unit.test_swebench_evaluate_predictions
```

That script runs the focused patch extraction, containerized-agent, and
prediction-evaluation unit tests without installing SWE-bench or running
Docker.

Generated files under `evals/out/` are local artifacts and are ignored by git.
Each subdirectory keeps a committed README describing the expected output layout
so that every user sees the same directory skeleton after cloning. See
[`evals/out/README.md`](out/README.md) for the full structure.

Record types:

- `trajectory`: raw messages/events/model turns for one run, written by scripts
- `eval_result`: pass/fail score and compact metrics, written by evals
- `training_example`: model-visible input plus assistant output, written by export scripts

## Adding a Containerized Suite

The generic framework lives in `simple_agent_lab.evals` (ADR 0017). It supplies
the container lifecycle, the Python/uv bootstrap, the run-directory convention,
and the one artifact seam, so a new Docker benchmark only implements what is
genuinely suite-specific:

1. A **host half** — a `Suite` (3 methods + 2 attributes), under
   `evals/<suite>/suite.py`:

   - `container_plan(instance)` -> `ContainerPlan` (image, workdir, shell,
     entrypoint as *data* — no `if`-branching in the runner).
   - `sanitize_instance(instance)` — drop gold/private fields before the agent
     sees the record.
   - `prediction_record(instance, *, model_name, result)` — shape the
     scorer-facing row from the container's result.
   - `name` and `container_module` (dotted path to the container half).

2. A **container half** — a module exposing `build_task(instance, *, workdir)`
   and `extract_result(workspace, instance)` (the "product", e.g. a `git diff`).
   Optional: `prepare(workspace, instance)` for pre-run setup (its return value
   is threaded back into `extract_result` as `context`), `agent_spec()` for the
   prompt/role and bash-vs-bash_task flavor, or `build_agent(...)` for full
   control. It **ships in the wheel** under
   `src/simple_agent_lab/evals/suites/<suite>/` (importing only stdlib + the
   installed runtime), so the in-container runner imports it with zero copying.

Then drive one instance. The same call runs locally or across machines — swap
the backend, not the code:

```python
from simple_agent_lab.evals import (
    run_suite_instance, LocalProcessBackend, LocalDockerBackend, LocalDirStore,
)

backend = LocalProcessBackend(workspace=ws)        # local dev: in-process, no Docker
# backend = LocalDockerBackend()                   # one machine, containerized
# backend = RemoteDockerBackend(base_url="tcp://worker:2375")  # remote daemon

run_suite_instance(
    suite=MySuite(),
    instance=instance,
    backend=backend,
    store=LocalDirStore(run_root),       # remote daemon: HostHttpStore (no S3) / S3Store
    run_root=Path("evals/out/mysuite"),
    run_id="run-1",
    provider_env={"OPENAI_MODEL": "...", "OPENAI_AUTH_TOKEN": "..."},
)
```

`LocalProcessBackend` runs the *exact* suite container half a container would
run, just in this process — fast iteration and a debugger during development —
then deploy containerized by swapping the backend. The execution environment is
the only real difference: a container's workspace + toolchain come from the
image; in-process you pass a local `workspace`. See
`examples/agent_judge/demo.py` for a candidate → agent-judge pipeline built this
way (plain Python over the shared store, no pipeline abstraction).

Inputs, the result, and the live trajectory all flow through the one `store`:
the host writes `input/instance.json`, the container writes `out/result.json`
and re-`put`s `out/trajectory.jsonl` on a cadence (the live trace), and the host
shapes `out/prediction.jsonl` from the result via `prediction_record`.

The same suite runs against a remote daemon by swapping `backend` / `store` —
both are `Protocol`s, so the suite and runner do not change. `HostHttpStore`
needs **no third-party middleware** (the host runs a stdlib HTTP server).
Reference: `evals/swebench/suite.py` (host half) +
`simple_agent_lab.evals.suites.swebench.container` (the two functions).
`FakeBackend` runs the whole flow without Docker for tests. Still follow the
output-directory checklist in [`out/README.md`](out/README.md).

### Local development → deployment (swap the backend)

A run has two orthogonal axes; you pick one value on each and change nothing
else:

| Axis | Option | When |
| --- | --- | --- |
| **Backend** (*where it runs*) | `LocalProcessBackend` | local dev: in-process, no Docker, debuggable |
| | `LocalDockerBackend` | one machine, in a container |
| | `RemoteDockerBackend` | remote daemon; **host-pull**, so the worker needs no reverse reachability |
| **Store** (*where bytes live*) | `LocalDirStore` | single machine (bind mount, zero-copy) |
| | `HostHttpStore` | remote daemon **when the worker can reach the host** (worker-push, no middleware) |
| | `S3Store` *(stub)* | host may go offline between submit and fetch |

**Matching backend to your network reality.** Multi-machine runs differ by which
way connections can be opened:

- **Worker can reach the host** (same LAN, host has an ingress): `HostHttpStore`
  — the worker pushes artifacts to a host-run stdlib HTTP server, no middleware.
- **Only the host can reach the worker** (worker behind NAT; host drives it via
  `DOCKER_HOST` / SSH — the common case): `RemoteDockerBackend`. The worker
  writes only to its own filesystem; the host moves bytes in/out with
  `put_archive` / `get_archive` over the *same outbound* host→worker connection
  it already uses to run the container. The worker needs **no** inbound
  reachability. Live trace is host-pull: pass `live_poll_interval_s=2` to have
  the host pull `out/trajectory.jsonl` on a cadence into local `evals/out/` for
  the viewer.
- **Neither can reach the other / host may go offline**: `S3Store` (stub) — both
  sides talk only to the bucket.

The fast inner loop while writing a suite is `LocalProcessBackend` + a fake
provider — it runs the *exact* container half, no image build:

```python
from simple_agent_lab.evals import (
    run_suite_instance, LocalProcessBackend, LocalDirStore,
)

art = run_suite_instance(
    suite=MySuite(),
    instance={"instance_id": "dev-1", ...},
    backend=LocalProcessBackend(workspace=my_local_workspace),
    store=LocalDirStore(run_root),
    run_root=run_root,
    run_id="dev",
    provider="fake",          # deterministic; no network. Use "openai" for a real model.
)
# art.run_dir / out/{result.json, trajectory.jsonl, prediction.jsonl}
```

Then deploy by changing one argument — `backend=LocalDockerBackend()` (and, for
a remote daemon, `store=HostHttpStore(run_root)`). The suite code does not move.
The only real difference is the workspace: a container gets it from the image
(`plan.workdir`); in-process you pass a local directory. Light suites and judges
run in-process directly; heavy suites (SWE-bench) still need their image.

### Running a whole dataset (`run_dataset`)

`run_suite_instance` runs one instance. `run_dataset` is the minimal
"controller" over it: it calls `run_suite_instance` once per instance on a
stdlib `ThreadPoolExecutor` (no Ray / asyncio / queue) and aggregates the
outcomes. The suite, backend, and store are unchanged — concurrency is just
"call the pure function N times in a pool," with the `ArtifactStore` as the
result bus.

```python
from simple_agent_lab.evals import run_dataset, LocalDockerBackend, LocalDirStore

report = run_dataset(
    suite=MySuite(),
    instances=dataset,                       # any iterable of instance dicts
    backend=LocalDockerBackend(),
    store=LocalDirStore(run_root),
    run_root=run_root,
    run_id="batch-1",
    concurrency=8,        # thread-pool size; default 1 = sequential
    max_attempts=2,       # retry only on a *raised* exception (infra/transient)
    on_result=lambda r: print(r.instance_id, "ok" if r.ok else r.error),
    provider="openai", provider_env={...},   # passed through to every run
)
report.summary()   # {"total": N, "ok": ..., "failed": ...}
```

Each instance lands under its own `<run_root>/<run_id>/<instance_id>/` tree, so
concurrent runs never collide and the viewer aggregates them into one batch.
`max_attempts` retries only raised exceptions (a daemon hiccup); a completed run
with a nonzero exit code is a result, not an error. Fan-out maps to backends:
Docker / remote backends give each run its own container, so `concurrency > 1`
is safe; a single `LocalProcessBackend(workspace=...)` shares one workspace, so
keep it sequential (or pass a workspace factory,
`workspace=lambda spec: base / spec.instance_id`, to fan out in-process safely).
This is the same worker-pool shape distributed frameworks (slime, ROLL) use for
rollout/reward workers — minus the RL-training machinery (GPU placement, weight
sync) that eval does not need.

### Long runs: submit now, reconcile later (host can leave)

`run_dataset` blocks until every instance finishes — fine for short or local
runs, but an agent can take minutes per instance, and you may not want to hold
the host process open for hours. With a backend whose work outlives the host (a
detached container — `LocalDockerBackend`), split it in two:

```python
from simple_agent_lab.evals import (
    submit_dataset, reconcile_dataset, LocalDockerBackend, LocalDirStore,
)

# 1. submit — start every container, write a manifest, return immediately
submit_dataset(
    suite=MySuite(), instances=dataset,
    backend=LocalDockerBackend(), store=LocalDirStore(run_root),
    run_root=run_root, run_id="batch-1",
    provider="openai", provider_env={...},
)
# ... the host may now exit / disconnect; the containers keep running ...

# 2. reconcile — a *fresh* process polls the batch to completion
report = reconcile_dataset(
    suite=MySuite(), backend=LocalDockerBackend(), store=LocalDirStore(run_root),
    run_root=run_root, run_id="batch-1", poll_interval_s=10,
)
report.summary()
```

How re-entry works: `submit_dataset` writes a manifest of serializable
`RunHandle`s to `<run_id>/batch.json` in the store — **re-persisted after each
container starts**, so a crash mid-submit still records every already-started
container (no orphans) — and each container writes its own `result.json` as it
finishes. `reconcile_dataset` reloads that manifest from the store (not from
memory), polls each handle, and shapes `prediction.jsonl` from each
`result.json`. It tolerates a daemon that no longer reports a container (already
collected, or polled by another process): a `result.json` on disk is taken as
the terminal truth, so reconcile never deadlocks and is safe to run repeatedly /
from a different machine. This is the eval-side of the "submit + poll" lifecycle
distributed frameworks use; only detaching backends provide `submit`/`poll`
(`LocalProcessBackend` cannot outlive the host and is run-only).

### Live trace + the trace viewer

The default runtimes — `LocalProcessBackend` or `LocalDockerBackend`, with
`LocalDirStore` — are compatible with the Observatory trace viewer
(`studio/trace-viewer`) and its live tail with no extra wiring, because the
framework preserves the viewer's three on-disk expectations (ADR 0016):

- **Layout** `<run_root>/<run_id>/<instance_id>/out/trajectory.jsonl` — the
  viewer parses `run_id` / `instance_id` from exactly this shape.
- **Filename** `trajectory.jsonl` (the `out/trajectory.jsonl` artifact key).
- **Schema** `simple-agent-lab.trajectory.v3`, written by
  `simple_agent_lab.trajectory.trace_record(...)`.

Live updates work because the in-container runner re-`put`s the trajectory key
on a cadence and `LocalDirStore` writes it atomically (`os.replace`), so a
polling viewer never reads a torn file; `meta.in_progress` flips to `false` on
the final write.

To watch a run live, point `run_root` under `evals/out/` (the viewer's default
scan dir) and start the viewer:

```bash
# run with run_root=Path("evals/out/mysuite"), then:
bash runs/run_trace_viewer.sh            # scans evals/out/
# or target a specific tree:
bash runs/run_trace_viewer.sh --dir evals/out/mysuite
```

The demo and tests use a throwaway `tempfile` dir to avoid leaving artifacts; use
`evals/out/<suite>/` when you actually want the viewer to see the run.

**Caveat — non-local stores.** The viewer scans a *local* directory, so it tails
runs only when artifacts land on the local filesystem: `LocalDirStore` (bind
mount) and `LocalProcessBackend` write straight to `evals/out/`, so live tail
works. With `HostHttpStore` / `S3Store` (remote daemon) the bytes live elsewhere;
point the viewer at wherever the host store persists them, or inspect after
`collect_outputs` brings them back. Local single-machine development — the common
path — is fully live.

### Composing runs (agent-as-judge, panels) without a pipeline engine

Because every run reads/writes through one `ArtifactStore`, a run is a node and
the store is the bus — so a multi-stage flow is just plain Python: run a
candidate, read its `result.json` / `trajectory.jsonl` from the store, feed them
as the instance of a second (judge) run. No pipeline abstraction. See
`examples/agent_judge/demo.py` for a candidate → agent-judge example that runs
two real agent loops in-process (`uv run python -m examples.agent_judge.demo`).

Evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Can we score model-call trajectories without scraping terminal output?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?
