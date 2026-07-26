# Multi-Machine Eval Deployment

Operations runbook for running the containerized eval framework across several
machines. Skip for single-machine work — use `LocalProcessBackend` /
`LocalDockerBackend` (see [`evals/README.md`](../evals/README.md)).

The package ships one `RemoteDockerBackend` per Docker daemon. A worker-pool
backend and a k8s backend are sketches below, **not** runnable APIs here.

## Workers run nothing of ours

A "worker" is just a machine running a Docker daemon. It runs no long-lived
agent and no installed copy of this project. The host drives each worker's
daemon remotely, and the container `pip install`s `simple-agent-lab` into an
isolated `/opt/agent-venv` at startup — never touching the image's own (conda)
environment that the benchmark depends on. So deploying a worker is two things:
**a daemon the host can reach, and the images in place.**

Why runtime injection instead of baking the agent into the image: agent code
iterates fast, benchmark images are large and stable, and SWE-bench Verified is
~500 per-instance images. Injection lets multi-machine runs use the *upstream*
images unchanged. The cost is one `pip install` per container start, removable
with a wheelhouse.

## What each worker needs

| Requirement | Satisfied by | Removable? |
| --- | --- | --- |
| Python + git in the container | the image (SWE-bench images ship both) | no |
| The agent wheel reachable | container egress to PyPI, or a mounted wheelhouse | yes — go offline with a wheelhouse |
| Outbound to the model API | the worker's network | **no — the agent must call the model** |
| host → worker Docker API | SSH or TLS | no |
| Images present | registry / `save`+`load` / Docker Hub | no |

## Reaching the daemon

The host must reach each worker's Docker API; workers never reach back — the
host-pull design depends only on host→worker, the direction usually available
even behind NAT.

- **SSH (recommended)** — `base_url="ssh://user@worker1"`. No extra port, reuses
  SSH auth and encryption.
- **TCP + TLS** — `base_url="tcp://worker1:2376"` with mutual-TLS certs. Lower
  per-call overhead, but you manage certificates.
- **Never** expose plain `tcp://worker:2375`: the daemon is root-equivalent, so
  an open 2375 hands worker root to anyone who can reach the port.

If neither direction is reachable, the host-driven model does not apply; you
would need a pull-based worker claiming jobs from a queue, which this project
does not implement.

## Getting images onto workers

SWE-bench is not one shared environment — each instance is its own image.

| Case | Strategy |
| --- | --- |
| SWE-bench **Pro** | published on Docker Hub; each worker pulls |
| **Verified**, few instances | build once, `docker save \| ssh worker docker load` |
| **Verified**, many / repeated | stand up an internal registry: build once, push, workers pull |

An internal registry (`docker run -p5000:5000 registry:2`) is the one piece of
shared infrastructure worth setting up at scale, and it doubles as a cache.
Under k8s it is mandatory, since pods schedule to arbitrary nodes.

## Online vs offline — decide before fanning out

Otherwise four machines all silently stall installing packages.

```text
Online (trusted intranet):   container reaches PyPI + the model API
                             → default pip install, nothing to pre-stage

Offline (recommended for scored runs):
                             container reaches ONLY the model API
                             → wheelhouse on each worker (--no-index --find-links)
                             → pre-pull images, run with --pull never
```

## Fanning out (sketch)

A pool backend would put several `RemoteDockerBackend`s behind one
`ContainerBackend`, capacity-limited per worker with a blocking slot queue.
Three properties then come free from the existing design: artifacts aggregate
in the host's one `run_root` (every worker is host-pull); the blocking queue
load-balances as "whoever is free takes the next one", routing around slow
machines; and a flaky worker just raises, so `run_dataset`'s `max_attempts`
retries onto a healthy worker with no failover code.

## Choosing a store by network reality

| Topology | Backend | Store |
| --- | --- | --- |
| host→worker only (NAT; common) | `RemoteDockerBackend` (host-pull) | host `LocalDirStore` |
| worker can reach host | same | `HostHttpStore` (worker push) |
| neither / k8s | remote or a future k8s backend | a future object-store `ArtifactStore` |

Kubernetes needs no change to `run_dataset`, the suites, or the stores: it is
one more `ContainerBackend`. What it changes is the store and image policy —
artifacts move through an object store rather than Docker host pulls, a registry
becomes mandatory, and submit/poll fits better than a blocking wait.
