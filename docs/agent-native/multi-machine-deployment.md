# Multi-Machine Eval Deployment

How to run the containerized eval framework across several machines. This is
an operations runbook for the backend and artifact-store seams described in
[`evals/README.md`](../../evals/README.md).

Read when: scaling an eval run beyond one machine, standing up workers, or
choosing a store/backend for a given network. Skip for single-machine local
development (use `LocalProcessBackend` / `LocalDockerBackend` — see
[`evals/README.md`](../../evals/README.md)).

The package currently ships one `RemoteDockerBackend` per Docker daemon.
`WorkerPoolBackend` and `K8sBackend` below are explicit extension sketches, not
runnable APIs in this repository.

## Mental model: workers run nothing of ours

A "worker" is **just a machine running a Docker daemon**. It does *not* run a
long-lived agent or any installed copy of this project. The host drives each
worker's daemon remotely (creates the container there) and the container
installs `simple-agent-lab` at startup (runtime injection — see below). So
"deploying a worker" is only two things: **(1) a Docker daemon the host can
reach, and (2) the images in place.**

```
host (scheduler; little compute)        workers (just a daemon + images)
  run_dataset(...)                         worker1   docker daemon
    └─ WorkerPoolBackend                   worker2   docker daemon
        ├─ RemoteDockerBackend ──API──▶    worker3   docker daemon
        └─ ...                             worker4   docker daemon
  artifacts pulled back to host run_root
```

## Runtime injection: we do not modify images

The agent code is **not built into the eval image**. The container's main
process is `python -m simple_agent_lab.evals.in_container`, and the bootstrap
script `pip install`s `simple-agent-lab` into an isolated `/opt/agent-venv` at
startup — never touching the image's own (conda) environment that the benchmark
under test depends on. `/testbed` stays exactly as the upstream image ships it;
the agent sees a normal local repo and a normal bash tool.

Why: agent code iterates fast, benchmark images are large and stable. Baking the
agent into ~500 per-instance images would mean rebuilding them on every agent
change. Runtime injection decouples the two, and lets multi-machine runs use the
**upstream** images unchanged (SWE-bench Pro from Docker Hub, Verified built
locally) instead of a private "modified" image.

The cost is one `pip install` per container start; eliminate it offline with a
wheelhouse (below).

## Per-worker requirements

| Requirement | Who satisfies it | Removable? |
| --- | --- | --- |
| Python inside the container | the image (SWE-bench images ship conda Python) | no — only missing on bare Alpine-style images |
| The agent wheel reachable | container reaches PyPI **or** a wheelhouse is mounted | yes — go offline with a wheelhouse |
| Outbound to the model API | the worker's network allows egress to `OPENAI_BASE_URL` | **no — the agent must call the model** |
| host → worker Docker API | host↔worker network (SSH or TLS) | no |
| `git` inside the container | the image (SWE-bench images ship git) | no (suite-specific: patch extraction) |
| Images present | each worker (registry / save-load / Docker Hub) | no |

The two requirements runtime injection adds are the first two rows. The one no
deployment can remove is model-API egress — the agent's whole job is to call the
model.

## Step 1 — expose each worker's Docker daemon to the host

The host must reach each worker's Docker API; the workers never need to reach
back (the framework's host-pull design depends only on host→worker, the
direction that is usually available even when workers sit behind NAT).

- **SSH (recommended).** `base_url="ssh://user@worker1"`. If you can `ssh` to the
  worker and it has Docker, you are done — no extra port, reusing SSH's auth and
  encryption. This is the simplest and safest form.
- **TCP + TLS.** `base_url="tcp://worker1:2376"` with mutual-TLS certs. Lower
  per-call overhead than SSH, but you manage certificates.
- **Never** expose plain `tcp://worker:2375` (no TLS): the Docker daemon is
  root-equivalent, so an open 2375 hands worker root to anyone who can reach the
  port.

If *neither* direction is reachable (fully sealed workers, host can't get in
either), the host-driven model does not apply; you would need a pull-based worker
(a long-lived agent on the worker that claims jobs from a queue / object store),
which this project does not implement. The host→worker SSH case covers the vast
majority of setups.

## Step 2 — get images onto every worker

SWE-bench is **not one shared environment**: each instance is its own image
(repo at the bug commit + its dependency stack), ~500 for Verified. There is no
single "environment" to download.

| Your case | Image strategy |
| --- | --- |
| SWE-bench **Pro** | images are published on Docker Hub — each worker `docker pull`s; nothing to build or host (workers just need egress) |
| **Verified**, few instances | build once, `docker save \| ssh worker docker load` to the others |
| **Verified**, many / repeated | stand up an internal registry: build once → `docker push` → workers `docker pull` |

Building compiles the repo's dependency stack (slow, CPU-heavy); pulling just
downloads pre-built layers (fast, and identical across machines). An internal
registry (`docker run -p5000:5000 registry:2`) is the one piece of shared
infrastructure worth setting up when you have many instances or many machines —
and it doubles as a cache: let the first run of an instance build+push, and all
later runs / machines pull. Under k8s a registry is not optional (pods schedule
to arbitrary nodes), so it becomes a prerequisite rather than an optimization.

## Step 3 — online vs offline (the wheel + image policy)

Two self-consistent configurations; decide which before fanning out, so four
machines don't all silently stall installing packages:

```
Online (simple; trusted intranet):
  container can reach PyPI + the model API
  → default pip install; nothing to pre-stage

Offline / clean (recommended for scored runs):
  container can reach ONLY the model API; PyPI blocked
  → stage a wheelhouse on each worker; bootstrap uses --no-index --find-links
  → pre-pull/load images locally; run with --pull never
```

Offline moves the "agent wheel reachable" requirement from "container egress" to
"a wheelhouse on each worker" — i.e. the wheelhouse is distributed to the four
machines just like the images.

## Step 4 — fan out across workers (extension sketch)

The host lists the workers and fans the dataset out over them. A pool backend
can combine several `RemoteDockerBackend`s behind one `ContainerBackend`;
conceptually:

```python
from simple_agent_lab.evals import run_dataset, LocalDirStore
# WorkerPoolBackend: a ContainerBackend that dispatches to a list of workers,
# capacity-limited per worker with a blocking slot queue.

backend = WorkerPoolBackend(
    [
        RemoteDockerBackend(base_url="ssh://user@worker1"),
        RemoteDockerBackend(base_url="ssh://user@worker2"),
        RemoteDockerBackend(base_url="ssh://user@worker3"),
        RemoteDockerBackend(base_url="ssh://user@worker4"),
    ],
    slots_per_backend=4,          # ≤4 concurrent containers per machine
)

report = run_dataset(
    suite=SwebenchSuite(),
    instances=dataset,
    backend=backend,
    store=LocalDirStore("evals/out/swebench"),   # host-pull lands all artifacts here
    run_root="evals/out/swebench",
    run_id="batch-1",
    concurrency=16,               # 4 machines × 4 slots
    provider="openai", provider_env={...},
)
report.summary()
```

Three properties fall out of the existing design, for free:

- **Artifact aggregation** — every worker is host-pull, so all
  `trajectory.jsonl` / `result.json` land in the host's one `run_root`; the trace
  viewer aggregates them as usual.
- **Load balancing** — the blocking slot queue is "whoever is free takes the next
  one," which beats round-robin and routes around slow machines; heterogeneous
  machines just get different `slots_per_backend`.
- **Failover** — a flaky worker raises, `run_dataset`'s `max_attempts` retries,
  and the retry takes a slot from a *healthy* worker. No failover code.

## Choosing a store by network reality

| Topology | Backend | Store |
| --- | --- | --- |
| host→worker only (NAT; the common case) | `RemoteDockerBackend` (host-pull) | host `LocalDirStore` (the aggregation point) |
| worker can reach host | same | `HostHttpStore` (worker push; no middleware) |
| neither / host may go offline / k8s | `RemoteDockerBackend` or `K8sBackend` | a future object-store `ArtifactStore` (S3/GCS) |

## Kubernetes (future)

k8s is compatible without touching `run_dataset` / the suites / the stores: it is
**one more `ContainerBackend`** (`K8sBackend.run(spec, store, binding)` submits a
Job built from `spec`, waits, returns a `RunOutcome`). What k8s changes is which
*store* and image policy you pair it with. Artifacts would move through an
object store or in-cluster HTTP store rather than Docker-specific host pulls; a
registry becomes mandatory, and a submit/poll lifecycle fits better than a
blocking wait. None of this requires changing the existing seams.
