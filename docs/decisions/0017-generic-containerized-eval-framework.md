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
the framework and is parameterized over three seams, each a `Protocol`:

1. **`ContainerBackend`** — where compute runs. `LocalDockerBackend` (today's
   docker-py behavior) ships first; `RemoteDockerBackend` / Kubernetes / managed
   runners can be added later without touching suites or the runner.
2. **`ArtifactTransport`** — how inputs reach the container and outputs come
   back. `BindMountTransport` (shared filesystem, today's behavior) ships first.
   `CopyOutTransport` (`put_archive` in, `get_archive` out — no shared filesystem
   required) is defined as a documented stub and is the path to cloud backends.
   `ObjectStoreTransport` (container uploads to S3/GCS) is noted as a future
   third implementation.
3. **`TraceSink`** — how live trace events leave the container. Live trace moves
   from "host tails a bind-mounted file" to **"container pushes records to a
   sink."** `FileTraceSink` (writes the same single-record `trajectory.jsonl`,
   behavior-preserving under a bind mount) ships first; `HttpTraceSink` /
   queue-based sinks are the cloud path and are defined as stubs.

The generic orchestration is one function, `run_suite_instance(...)`, that wires
a `Suite` + `ContainerBackend` + `ArtifactTransport` + `TraceSink` together. It
contains no `if pro:` branches and no Docker calls of its own.

### Suite split: host half and container half

A containerized eval is fundamentally two programs. The split is made explicit:

- **Host half** (`Suite`, runs where the orchestrator runs): resolve the image
  and launch shape (`container_plan`), drop gold/private fields
  (`sanitize_instance`), and shape the prediction record (`prediction_record`).
- **Container half** (referenced by `Suite.container_module`, runs inside the
  image): build the model-visible task from the instance, and **extract the
  result** — the "product" of the run (for SWE-bench, the `git diff`). The
  generic in-container runner owns the agent loop, retry, and trace push; the
  suite module only supplies `build_task(...)` and `extract_result(...)`.

This keeps result extraction next to the workspace it inspects (inside the
container) while keeping the agent loop generic.

### Scope of this change

This ADR lands the framework end to end as the **integration contract**:

- the protocols, `LocalDockerBackend`, `BindMountTransport`, `FileTraceSink`,
  an in-memory `FakeBackend`, and `run_suite_instance` (host side);
- the **generic in-container runner** (`simple_agent_lab.evals.in_container`):
  it imports a suite's `container_module`, builds the agent, drives the loop
  with retry, pushes the trajectory to a `TraceSink`, and writes the raw
  `extract_result` product to ``result.json``; the host shapes
  ``prediction.jsonl`` from it via `prediction_record`.

SWE-bench is the reference suite: `evals/swebench/suite.py` (host half) plus
`evals/swebench/container.py` (the two functions + optional `prepare` /
`agent_spec`). A new fake-backend test composes the seams without Docker, and a
second test drives the SWE-bench container half through the generic runner with
the fake provider on a hermetic git repo — so the "one Suite + two functions"
shape is exercised, not just asserted.

What remains is **retiring the legacy launcher**: pointing the production run
scripts at `run_suite_instance` + the generic runner and deleting the
`is_swebench_pro` branches from `containerized_agent.py` / `in_container_runner.py`
(today the container half re-uses their task text and patch helpers as the
single source of truth). `evaluate_predictions.py` (scoring) stays untouched.
`CopyOutTransport` and `HttpTraceSink` remain documented stubs until a concrete
cloud backend lands.

## Consequences

- A new Docker benchmark is "implement `Suite` + a container module," not "copy
  `containerized_agent.py` and edit the branches."
- The cloud story has explicit seams: switching to a remote daemon is choosing
  `RemoteDockerBackend` + `CopyOutTransport` + a non-file `TraceSink`, with the
  suite and runner unchanged.
- ADR 0011's "adapter, not framework" guidance is **superseded** for the
  containerized case. Its core principles survive: the runtime core still knows
  nothing about Docker, datasets, or gold patches; raw trajectories remain
  reusable when scoring rules change; scoring stays a separate path.
- Short-term cost: two ways to launch SWE-bench coexist until the cutover lands.
  The skeleton is import-clean and unit-tested, but the production run scripts
  keep calling the existing launcher until the follow-up PR flips them.

## Alternatives Considered

- **Keep extending the `is_pro` branches.** Rejected — the branch count grows
  per suite and the cloud transport change would have to thread through every
  branch.
- **One `EvalSuite` base class with overridable methods.** Rejected — conflicts
  with the project's "config is data, behavior is module-level functions /
  injected implementations" taste (see `llm.Provider`).
- **Make the transport always copy-out (drop bind mount).** Rejected for now —
  bind mount gives zero-copy artifacts and the lowest-latency live trace locally;
  it stays the default while `CopyOutTransport` is the opt-in for remote daemons.
- **Abstract the cloud backend now.** Deferred — only the seam (`ContainerBackend`
  / `ArtifactTransport` protocols) is defined now; the remote implementation
  waits for a concrete cloud target, per "learn before abstracting."
