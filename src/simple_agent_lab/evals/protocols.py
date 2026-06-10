"""Protocols and data values for the generic containerized eval framework.

The shapes here mirror `llm.provider`'s taste: configuration is *data*
(frozen dataclasses, JSON-friendly), and capability is a `Protocol` that a
small concrete implementation satisfies — no class hierarchy to subclass.

Two orthogonal seams keep the runner portable from a local Docker daemon to a
cloud backend without touching suites or `run_suite_instance`:

- `ContainerBackend` — *where* compute runs (local docker-py today; remote /
  k8s later).
- `ArtifactStore` — *where bytes live*: one keyed store that carries inputs,
  the result, and the live trajectory, in both directions. `LocalDirStore`
  (bind mount) and `HostHttpStore` (a batteries-included stdlib server, no
  third-party middleware) ship today; an object store (S3/GCS) is a future
  addition behind the same protocol.

There is deliberately no separate "transport" or "trace sink": staging inputs,
collecting outputs, and pushing the live trace are all just `put`/`get` on the
one `ArtifactStore`. The live trajectory is simply an artifact key that gets
re-`put` on a cadence.

A `Suite` supplies only the suite-specific bits. Its *host half* (launch shape
and the agent-visible task input) lives behind this
protocol; its *container half* — `build_task` / `extract_result` — is referenced
by `container_module` and imported by the in-container runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..messages import ContentInput

# Fixed artifact keys the framework and the in-container runner agree on. Keys
# are relative to one instance's run directory, so a store bound to that dir
# (or a bind mount of it) resolves them identically on host and container.
INSTANCE_KEY = "input/instance.json"  # host puts (sanitized), container gets
EVAL_KEY = (
    "input/eval.json"  # host puts (gold scoring inputs), container `evaluate` gets
)
RESULT_KEY = "out/result.json"  # container puts (raw extract_result), host gets
TRACE_KEY = "out/trajectory.jsonl"  # container re-puts on a cadence = live trace

# Optional persistent-memory contract for containerized eval runs. The memory
# package remains local-filesystem-only; container backends may bind-mount a host
# directory and point in-container agent assembly at it with this env var.
MEMORY_HOME_ENV = "SAL_MEMORY_HOME"
DEFAULT_MEMORY_CONTAINER_HOME = "/agent/memory"
# Optional in-container memory namespace + run id. The host sets these so the
# in-container assembly can name the FilesystemMemory namespace (e.g. the repo)
# and the per-run evidence directory without teaching memory about the suite.
MEMORY_NAME_ENV = "SAL_MEMORY_NAME"
MEMORY_RUN_ID_ENV = "SAL_MEMORY_RUN_ID"


@dataclass(frozen=True)
class LaunchSpec:
    """Suite-resolved, backend-agnostic launch shape for one instance.

    Captures the per-instance launch values that would otherwise make the runner
    fork on suite-specific conditions, so a suite expresses them as data instead
    of the runner branching on them.

    - `entrypoint`: `""` clears an image `ENTRYPOINT` (some images set one);
      `None` keeps the image default.
    - `shell`: the argv prefix the bootstrap script is passed to, e.g.
      `("bash", "-lc")` vs `("/bin/sh", "-lc")`.
    - `cap_add`: extra Linux capabilities for the container (docker `--cap-add`).
      Docker drops most caps by default; some suites' test steps need specific
      ones (e.g. `SYS_PTRACE`, used by ptrace/strace/gdb-style tooling). A suite
      resolves the set its image requires; the default `()` adds none.
    - `network_mode`: passed straight through to Docker (`--network`); `None`
      leaves it unset (Docker's default `bridge`). Common values: `"host"`,
      `"none"` (offline/sandboxed), `"bridge"`, `"container:<name|id>"`, or a
      custom network name. Not validated here — bad values fail at create time.
    """

    image: str
    workdir: str
    shell: tuple[str, ...] = ("bash", "-lc")
    entrypoint: str | None = None
    platform: str | None = None
    cap_add: tuple[str, ...] = ()
    network_mode: str | None = None


@dataclass(frozen=True)
class ContainerBinding:
    """What a store needs the backend to apply so the container sees its keys.

    `mounts` are docker-style bind mounts (used by `LocalDirStore`); `env` are
    environment variables the in-container runner reads to reconstruct the
    container-side store (used by `HostHttpStore` / a future object store). A store returns
    one or the other (or both); the backend applies whatever is present.
    `add_hosts` maps hostnames to IPs (e.g. ``host-gateway``) so a container can
    reach a host-run store on Linux.
    """

    mounts: dict[str, dict[str, str]] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    add_hosts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifacts:
    """What one instance run produced. Pure data; safe to log.

    The run phase produces the raw `extract_result` product (``out/result.json``)
    and the trajectory. A suite that scores in the run environment (the optional
    container-half ``evaluate`` hook) merges its verdict into the same
    ``out/result.json``; otherwise scoring is a follow-up run (an agent judge) or
    an external oracle (the official harness).
    """

    instance_id: str
    run_dir: Path
    trajectory_path: Path
    status_code: int
    logs: str = ""


@dataclass(frozen=True)
class AgentSpec:
    """How the in-container runner should build the agent for a suite.

    A container module may expose ``agent_spec()`` returning this; otherwise the
    runner uses the defaults (a plain bash agent with no system prompt). A suite
    that needs full control can instead expose ``build_agent(...)`` directly, so
    the framework never enumerates every agent shape. Only the prompt/role/flavor
    are suite-specific — the loop, retry, and trace push are generic.
    """

    name: str = "agent"
    role: str = ""
    system_prompt: str = ""
    flavor: str = "bash"  # "bash" | "bash_task" | "bash_skills"


@runtime_checkable
class Suite(Protocol):
    """Suite-specific host half. Everything else is provided by the framework.

    `container_module` is the dotted import path of the suite's container half
    (a module exposing `build_task(instance, *, workdir)` and
    `extract_result(workspace, instance)`, plus optional `prepare` /
    `agent_spec` / `build_agent`). The generic in-container runner
    (`simple_agent_lab.evals.in_container`) imports it so the agent loop, retry,
    and trace push stay suite-agnostic. The module must import only the standard
    library and the installed ``simple-agent-lab`` wheel, since it runs inside
    the image.
    """

    name: str
    container_module: str

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec: ...

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """The agent-visible task input: gold/private fields dropped before the
        record is shown to the agent. The host stages it under
        ``input/instance.json`` (INSTANCE_KEY); the gold counterpart the agent
        must not see goes through the optional ``eval_inputs``."""
        ...

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Gold/private data the in-environment ``evaluate`` hook needs (e.g. the
        official eval script + expected tests, or the expected answer) but the
        agent must not see. The host stages it under ``input/eval.json``
        (EVAL_KEY) — separate from the agent-visible ``input/instance.json`` —
        and the generic runner threads it into the container-half ``evaluate``
        hook as ``context["eval"]``, which is what turns the hook on. Return
        ``None`` to score elsewhere (a follow-up agent-judge run or the official
        harness) instead of in the run environment."""
        ...


@runtime_checkable
class ContainerTask(Protocol):
    """The container half a suite module supplies (runs inside the image).

    Documentation of the duck-typed surface the in-container runner imports by
    `Suite.container_module`. ``build_task`` and ``extract_result`` are the two
    functions a new suite must write; ``prepare`` (pre-run setup, threaded into
    ``extract_result`` as ``context``), ``apply_oracle`` (apply the reference
    solution for the model-free oracle self-check), ``evaluate`` (score in the
    run environment — see below), ``memory_artifacts`` (collect this run's
    durable memory products from the workspace — see below), ``agent_spec``, and
    ``build_agent`` are optional.

    Optional ``evaluate(workspace, instance, *, context)`` scores in the run
    environment: after ``extract_result``, the generic runner calls it when the
    suite staged ``eval_inputs`` (available as ``context["eval"]``) and merges
    its returned verdict into ``out/result.json``. It is environment-neutral —
    the generic runner is driven by both `LocalProcessBackend` (in-process) and
    container backends, so ``evaluate`` runs wherever the run ran, with no Docker
    assumption. A suite that scores elsewhere (a follow-up agent-judge run or the
    official harness) simply omits it.

    Optional ``memory_artifacts(workspace, instance, *, context)`` returns this
    run's durable products as memory ``FilesystemArtifact`` values gathered
    straight from the workspace — the same ``(workspace, instance, *, context)``
    shape as ``extract_result``. When persistent memory is active the generic
    runner injects it as the memory ``artifact_builder``, so the products are
    captured inside ``memory.finish`` at the standard ``SESSION_END`` hook —
    while the workspace is still intact, before ``extract_result``. The hook name
    is generic: any suite can register its own collector, and a suite that does
    not integrate with memory simply omits it (memory then falls back to its
    generic defaults). Keeping this a suite-supplied hook is what keeps the
    runner free of any suite-specific product flow.
    """

    def build_task(self, instance: Mapping[str, Any], *, workdir: str) -> ContentInput:
        """Build the model-visible task: `str`, or content blocks for multimodal.

        Returning a sequence of content blocks (text + `ImageBlock`) lets a suite
        feed images alongside text; a plain `str` is the common case and is
        normalized to a single text block.
        """
        ...

    def extract_result(
        self,
        workspace: Any,
        instance: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return the run's raw product (e.g. ``{"model_patch": diff}``).

        Accept ``context`` (keyword) to receive whatever an optional
        ``prepare(workspace, instance)`` returned — e.g. a baseline commit. The
        in-container runner only passes ``context`` when the signature declares
        it, but declaring it is what lets pre-run setup reach extraction; omit it
        and a `prepare` step's output is silently dropped.
        """
        ...


@dataclass(frozen=True)
class RunSpec:
    """Structured, backend-agnostic description of one run.

    The same spec drives every backend: `LocalDockerBackend` turns it into a
    container command, `LocalProcessBackend` runs it in the current process, a
    remote backend ships it elsewhere. The backend reads the instance and writes
    the result/trajectory through the bound `ArtifactStore` it is handed, so the
    suite's container half runs identically in or out of a container.
    """

    suite_name: str
    container_module: str
    instance_id: str
    launch_spec: LaunchSpec
    max_turns: int
    provider: str  # "openai" | "fake"
    api_kind: str
    provider_env: Mapping[str, str] = field(default_factory=dict)
    install: bool = True
    wheelhouse_mount: str | None = None
    run_name: str = ""


@dataclass(frozen=True)
class RunOutcome:
    """What a backend reports back after one run."""

    status_code: int
    logs: str = ""


@dataclass(frozen=True)
class RunHandle:
    """A serializable reference to a submitted run, for host-reentrant polling.

    Detached container runs keep going after the submitting host process exits,
    so a long batch can be submitted, the host can leave, and a *new* host
    process can re-attach by reloading these handles from the store and polling.
    The handle carries only plain data (no live objects), so it round-trips
    through JSON in the `ArtifactStore`.

    `ref` is the backend-specific locator (e.g. a container name); `extra` holds
    any other reconnection data a backend needs (e.g. a remote daemon URL).
    """

    backend_kind: str
    ref: str
    run_dir: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContainerBackend(Protocol):
    """Where a run executes. The backend owns the full lifecycle.

    `LocalDockerBackend` launches a container; `LocalProcessBackend` runs the
    agent loop in the current process (local dev, no Docker); a remote backend
    ships the spec to another machine. All consume the same `RunSpec` and the
    same bound `ArtifactStore`, so swapping the backend — not the code — is what
    moves a suite from local development to multi-machine deployment.

    `run` is the blocking lifecycle (create → wait → collect → remove) used by
    `run_suite_instance` / `run_dataset`. Backends whose work outlives the host
    process (detached containers) may *also* implement the optional `submit` /
    `poll` pair, enabling host-reentrant batches (`submit_dataset` /
    `reconcile_dataset`): `submit` starts the run and returns a serializable
    `RunHandle` without waiting; `poll` reports a `RunOutcome` once it finishes
    (or `None` while still running). Backends that cannot outlive the host
    (`LocalProcessBackend`) implement only `run`.
    """

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome: ...

    # Optional (duck-typed; not all backends provide these):
    #   def submit(self, spec, *, store, binding) -> RunHandle: ...
    #   def poll(self, handle: RunHandle) -> RunOutcome | None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """One keyed byte store shared between host and container, both directions.

    Subsumes input staging, output collection, and live-trace push. `bind`
    returns a view rooted at one instance's run directory so a single host-side
    store (e.g. one `HostHttpStore` server) can serve many instances. The
    container reconstructs its own store from `container_binding().env`; it never
    calls `bind`.
    """

    def bind(self, run_dir: Path) -> ArtifactStore:
        """Return a per-instance view rooted at `run_dir`."""
        ...

    def get(self, key: str) -> bytes: ...

    def put(self, key: str, data: bytes) -> None: ...

    def container_binding(self) -> ContainerBinding:
        """Mounts/env the backend must apply so the container resolves keys."""
        ...

    def collect_outputs(self) -> None:
        """Host-side: ensure outputs are on local disk (no-op for shared-FS stores)."""
        ...
