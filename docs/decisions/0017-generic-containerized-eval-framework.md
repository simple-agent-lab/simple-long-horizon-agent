# ADR 0017: Generic Containerized Eval Framework

## Status

Proposed

## Context

ADR 0011 decided to represent benchmark integrations as suite-specific adapters
under `evals/` and explicitly warned against building a framework, a registry, or
a plugin loader "before the second or third suite proves the need."

That need has now arrived inside a single adapter. `evals/swebench/` already
serves **two** suites — SWE-bench Verified and SWE-bench Pro — and they diverge
through `is_swebench_pro_instance(...)` branches scattered across the host
launcher:

- `docker_image_for_instance` — local `make_test_spec` image vs Docker Hub image.
- `resolve_workdir` — `/testbed` vs `/app`.
- `docker_run_command` / `container_entrypoint_override` — `bash -lc` vs
  `/bin/sh -lc`, and whether to clear the image `ENTRYPOINT`.
- `prediction_record` — `{model_patch, model_name_or_path}` vs `{patch, prefix}`.

Meanwhile the bulk of `containerized_agent.py` and `in_container_runner.py` is
**suite-agnostic** and is what every future Docker benchmark would re-implement:
container lifecycle, file copy-in, the Python/uv bootstrap script, wheelhouse
preparation, the run-directory convention, instance JSON/JSONL loading
(duplicated verbatim across both files), credential env passthrough, the agent
loop + rate-limit retry, and trajectory writing.

A second, sharper constraint now drives the design: **the Docker daemon may live
in the cloud, not on the host.** Today artifacts come back through a bind mount
(`volumes={run_dir: {"bind": "/agent/run"}}`), which silently assumes the daemon
and the orchestrating process share one filesystem. That assumption breaks the
moment Docker runs over `DOCKER_HOST=tcp://…`, in Kubernetes, or behind a managed
container service. The bind mount is the single biggest blocker to running the
same eval against a remote/cloud backend.

## Decision

Promote the suite-agnostic machinery into a small, teachable package,
`src/simple_agent_lab/evals/`, structured as **data + protocols + small
functions** — the same taste as `llm.Provider` (config is data; capability is a
registered/injected implementation, not a class hierarchy).

A benchmark suite becomes a thin implementation of the `Suite` protocol that
supplies only what is genuinely suite-specific. Everything else is provided by
the framework and is parameterized over **two orthogonal seams**, each a
`Protocol`:

1. **`ContainerBackend`** — *where a run executes*. It receives a structured
   `RunSpec` + the bound `ArtifactStore` and owns the whole lifecycle, so the
   same call runs anywhere by swapping the backend:
   - `LocalProcessBackend` — runs the agent loop in the current process, no
     Docker. This is the local-development path: the *exact* suite container
     half a container would run, but with a debugger and instant iteration.
   - `LocalDockerBackend` — turns the `RunSpec` into a container command (the
     in-wheel generic runner) and runs it on the local / `DOCKER_HOST` daemon.
   - `RemoteDockerBackend` — a remote daemon with **host-pull** artifact
     movement: the worker writes only to its own filesystem and the host moves
     bytes in/out with `put_archive` / `get_archive` over the same outbound
     host→worker connection. This fits the common topology where the host can
     reach the worker but the worker (behind NAT) cannot reach back — the case
     `HostHttpStore` (worker-push) does *not* cover.
   - Kubernetes / managed runners — later, unchanged suites. Swapping the
     backend (not the code) is what moves a suite from local development to
     multi-machine deployment.

   The execution *environment* is the deliberate difference, not the code: a
   container gets its workspace + toolchain from the image (`plan.workdir`);
   in-process you supply a local workspace. Light suites and judges run
   in-process directly; heavy suites (SWE-bench) still need their image.
2. **`ArtifactStore`** — *where bytes live*. One keyed store carries inputs,
   the result, and the live trajectory, in both directions. There is
   deliberately **no separate transport or trace sink**: staging inputs,
   collecting outputs, and pushing the live trace are all just `put`/`get`, and
   the live trajectory is simply an artifact key re-`put` on a cadence. Three
   backends span the spectrum:
   - `LocalDirStore` — bind mount, single machine, zero moving parts.
   - `HostHttpStore` — the host runs a stdlib `http.server` over the run
     directory; the container reads/writes over HTTP. This is "bind mount over
     the network": it works across a **remote daemon with no third-party
     middleware**, which keeps the teaching path runnable out of the box.
   - `S3Store` — object store for fully decoupled runs (documented stub).

   "Container pushes its live trace" is preserved: it is just `put` of the
   trajectory key (a file write under `LocalDirStore`, an HTTP POST under
   `HostHttpStore`).

The generic orchestration is one function, `run_suite_instance(...)`, that wires
a `Suite` + `ContainerBackend` + `ArtifactStore` together, **builds the container
command itself** (bootstrap + `python -m simple_agent_lab.evals.in_container`, in
the wheel — nothing is copied in), and contains no `if pro:` branches and no
Docker calls of its own.

### Suite split: host half and container half

A containerized eval is fundamentally two programs. The split is made explicit:

- **Host half** (`Suite`, runs where the orchestrator runs): resolve the image
  and launch shape (`container_plan`), drop gold/private fields
  (`sanitize_instance`), and shape the prediction record (`prediction_record`).
  Lives next to the suite's adapter under `evals/<suite>/`, where heavy host-only
  deps (docker, the official harness) stay out of the core package.
- **Container half** (referenced by `Suite.container_module`, runs inside the
  image): `build_task` + `extract_result` (plus optional `prepare` /
  `agent_spec` / `build_agent`). It **ships in the wheel** under
  `simple_agent_lab.evals.suites.<suite>/`, so the in-container runner imports it
  with zero file copying and never drifts from the installed runtime. It must
  import only the standard library and the installed wheel. The dependency-
  isolation reason suites live under `evals/` applies to the heavy *host* half,
  not this light container half.

The generic in-container runner owns the agent loop, retry, and trace push; it
writes the raw `extract_result` product to `out/result.json` via the store, and
the host shapes `out/prediction.jsonl` from it — so prediction formatting stays
host-side and works unchanged under any store backend.

### Scope of this change

This ADR lands the framework end to end as the **integration contract**:

- the protocols, `LocalDockerBackend`, `LocalDirStore`, `HostHttpStore`, an
  in-memory `FakeBackend`, and `run_suite_instance` (host side);
- the **generic in-container runner** (`simple_agent_lab.evals.in_container`):
  it imports a suite's `container_module`, builds the agent, drives the loop
  with retry, re-`put`s the trajectory to the store on a cadence, and writes the
  raw `extract_result` product to `out/result.json`; the host shapes
  `out/prediction.jsonl` from it via `prediction_record`.

SWE-bench is the reference suite: `evals/swebench/suite.py` (host half) plus the
in-wheel `simple_agent_lab.evals.suites.swebench.container` (the two functions +
optional `prepare` / `agent_spec`). Tests, all Docker-free: the demo suite runs
through `run_suite_instance` against `FakeBackend` over **both** `LocalDirStore`
and `HostHttpStore` (the latter over real loopback HTTP, proving the
batteries-included store), and the SWE-bench container half runs through the
generic runner with the fake provider on a hermetic git repo — so the "one Suite
+ two functions" shape is exercised, not just asserted.

What remains is **retiring the legacy launcher**: pointing the production run
scripts at `run_suite_instance` + the generic runner and deleting the
`is_swebench_pro` branches from `containerized_agent.py` / `in_container_runner.py`.
The patch helpers are already the single source of truth in the wheel
(`evals/swebench/patch_extract.py` is now a re-export shim); the SWE-bench task
text is briefly duplicated in the container half until the legacy runner is
deleted. `evaluate_predictions.py` (scoring) stays untouched. `S3Store` remains
a documented stub until a concrete cloud target lands.

## Consequences

- A new Docker benchmark is "implement `Suite` + a container module," not "copy
  `containerized_agent.py` and edit the branches."
- The cloud story has explicit seams along two orthogonal axes, chosen by which
  way connections open: worker→host reachable → `HostHttpStore` (worker-push, no
  middleware); only host→worker reachable (NAT, the common case) →
  `RemoteDockerBackend` (host-pull, worker needs no inbound reachability);
  neither / host may go offline → `S3Store`. The suite and runner are unchanged
  across all three.
- ADR 0011's "adapter, not framework" guidance is **superseded** for the
  containerized case. Its core principles survive: the runtime core still knows
  nothing about Docker, datasets, or gold patches; raw trajectories remain
  reusable when scoring rules change; scoring stays a separate path.
- The core package now ships suite *container halves* (light, stdlib-only) under
  `src/simple_agent_lab/evals/suites/`. Heavy host halves stay under `evals/`. A
  suite therefore spans two trees; the host `suite.py` remains the readable
  entry point.
- Short-term cost: two ways to launch SWE-bench coexist until the cutover lands.
  The new path is import-clean and unit-tested; the production run scripts keep
  calling the existing launcher until the follow-up PR flips them.

## Alternatives Considered

- **Keep extending the `is_pro` branches.** Rejected — the branch count grows
  per suite and the cloud storage change would have to thread through every
  branch.
- **One `EvalSuite` base class with overridable methods.** Rejected — conflicts
  with the project's "config is data, behavior is module-level functions /
  injected implementations" taste (see `llm.Provider`).
- **Keep `ArtifactTransport` + `TraceSink` as separate seams.** Rejected — they
  answer the same question ("how do bytes move host↔container") on one axis;
  collapsing them into one `ArtifactStore` removes a whole concept and makes the
  storage backend (`LocalDir` / `HostHttp` / `S3`) a single implementation point
  instead of two.
- **Make every store copy bytes (drop bind mount).** Rejected — `LocalDirStore`
  bind mount gives zero-copy artifacts and the lowest-latency live trace
  locally; it stays the default while `HostHttpStore` is the no-middleware
  opt-in for remote daemons.
- **Ship suite container halves only under `evals/` and copy them in.** Rejected
  — copying the import closure reintroduces version skew with the installed
  wheel; the light container half is safe to ship, and the `container_module`
  string still allows out-of-tree modules later.
- **Abstract the cloud backend now.** Deferred — only the seam (`ContainerBackend`
  / `ArtifactStore` protocols) is defined now; the remote implementation
  waits for a concrete cloud target, per "learn before abstracting."
