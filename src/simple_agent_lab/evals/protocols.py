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
  third-party middleware) ship today; `S3Store` is the production stub.

There is deliberately no separate "transport" or "trace sink": staging inputs,
collecting outputs, and pushing the live trace are all just `put`/`get` on the
one `ArtifactStore`. The live trajectory is simply an artifact key that gets
re-`put` on a cadence.

A `Suite` supplies only the suite-specific bits. Its *host half* (image and
launch shape, instance sanitization, prediction shape) lives behind this
protocol; its *container half* — `build_task` / `extract_result` — is referenced
by `container_module` and imported by the in-container runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Fixed artifact keys the framework and the in-container runner agree on. Keys
# are relative to one instance's run directory, so a store bound to that dir
# (or a bind mount of it) resolves them identically on host and container.
INSTANCE_KEY = "input/instance.json"  # host puts, container gets
RESULT_KEY = "out/result.json"  # container puts (raw extract_result), host gets
TRACE_KEY = "out/trajectory.jsonl"  # container re-puts on a cadence = live trace


@dataclass(frozen=True)
class ContainerPlan:
    """Suite-resolved, backend-agnostic launch shape for one instance.

    Captures exactly the values the SWE-bench launcher currently forks on via
    `is_swebench_pro_instance(...)`, so a suite expresses them as data instead
    of the runner branching on them.

    - `entrypoint`: `""` clears an image `ENTRYPOINT` (Pro images set one);
      `None` keeps the image default.
    - `shell`: the argv prefix the bootstrap script is passed to, e.g.
      `("bash", "-lc")` vs `("/bin/sh", "-lc")`.
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
    container-side store (used by `HostHttpStore` / `S3Store`). A store returns
    one or the other (or both); the backend applies whatever is present.
    `add_hosts` maps hostnames to IPs (e.g. ``host-gateway``) so a container can
    reach a host-run store on Linux.
    """

    mounts: dict[str, dict[str, str]] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    add_hosts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifacts:
    """What one instance run produced. Pure data; safe to log."""

    instance_id: str
    run_dir: Path
    trajectory_path: Path
    prediction_path: Path
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
    flavor: str = "bash"  # "bash" | "bash_task"


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

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan: ...

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        """Drop gold/private fields before the record is shown to the agent."""
        ...

    def prediction_record(
        self,
        instance: Mapping[str, Any],
        *,
        model_name: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Shape the suite's scorer-facing prediction row from `extract_result`."""
        ...


@runtime_checkable
class ContainerTask(Protocol):
    """The container half a suite module supplies (runs inside the image).

    Documentation of the duck-typed surface the in-container runner imports by
    `Suite.container_module`. ``build_task`` and ``extract_result`` are the two
    functions a new suite must write; ``prepare`` (pre-run setup, threaded into
    ``extract_result`` as ``context``), ``agent_spec``, and ``build_agent`` are
    optional.
    """

    def build_task(self, instance: Mapping[str, Any], *, workdir: str) -> str: ...

    def extract_result(
        self, workspace: Any, instance: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Return the run's raw product (e.g. ``{"model_patch": diff}``)."""
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
    plan: ContainerPlan
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


@runtime_checkable
class ContainerBackend(Protocol):
    """Where a run executes. The backend owns the full lifecycle.

    `LocalDockerBackend` launches a container; `LocalProcessBackend` runs the
    agent loop in the current process (local dev, no Docker); a remote backend
    ships the spec to another machine. All consume the same `RunSpec` and the
    same bound `ArtifactStore`, so swapping the backend — not the code — is what
    moves a suite from local development to multi-machine deployment.
    """

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome: ...


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
