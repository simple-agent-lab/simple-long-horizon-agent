"""Protocols and data values for the generic containerized eval framework.

The shapes here mirror `llm.provider`'s taste: configuration is *data*
(frozen dataclasses, JSON-friendly), and capability is a `Protocol` that a
small concrete implementation satisfies — no class hierarchy to subclass.

Three seams keep the runner portable from a local Docker daemon to a cloud
backend without touching suites or `run_suite_instance`:

- `ContainerBackend` — *where* compute runs (local docker-py today).
- `ArtifactTransport` — *how* inputs reach the container and outputs return
  (bind mount today; copy-out for remote daemons).
- `TraceSink` — *how* live trace records leave the container (file today;
  HTTP/queue for cloud).

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
class RunArtifacts:
    """What one instance run produced. Pure data; safe to log."""

    instance_id: str
    run_dir: Path
    trajectory_path: Path
    prediction_path: Path
    status_code: int
    logs: str = ""


@dataclass(frozen=True)
class StagedFile:
    """One file the transport must make readable inside the container."""

    data: bytes
    container_path: str
    mode: int = 0o644


@runtime_checkable
class Suite(Protocol):
    """Suite-specific host half. Everything else is provided by the framework.

    `container_module` is the dotted import path of the suite's container half
    (a module exposing `build_task(instance, *, workdir)` and
    `extract_result(workspace, instance)`); the generic in-container runner
    imports it so the agent loop, retry, and trace push stay suite-agnostic.
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
class ContainerHandle(Protocol):
    """One created-but-not-removed container, abstracted over the backend."""

    def put_file(self, file: StagedFile) -> None:
        """Place one file inside the container before it starts."""
        ...

    def start(self) -> None: ...

    def wait(self) -> int:
        """Block until the main process exits; return its status code."""
        ...

    def logs(self) -> str: ...

    def get_archive(self, container_path: str) -> bytes:
        """Return a tar stream of `container_path` (used by copy-out transport)."""
        ...

    def remove(self) -> None: ...


@runtime_checkable
class ContainerBackend(Protocol):
    """Where containers run. `LocalDockerBackend` is the only implementation today."""

    def create(
        self,
        *,
        name: str,
        plan: ContainerPlan,
        command: tuple[str, ...],
        env: Mapping[str, str],
        mounts: Mapping[str, Mapping[str, str]],
    ) -> ContainerHandle: ...


@runtime_checkable
class ArtifactTransport(Protocol):
    """How inputs reach the container and outputs come back.

    `BindMountTransport` returns bind mounts and relies on the shared
    filesystem; `CopyOutTransport` returns no mounts and instead copies inputs
    in / outputs out over the backend's archive API, so it works against a
    remote daemon.
    """

    def mounts(self, run_dir: Path) -> dict[str, dict[str, str]]:
        """Bind mounts to request at create time (empty for copy-out)."""
        ...

    def stage_inputs(
        self,
        handle: ContainerHandle,
        *,
        run_dir: Path,
        files: tuple[StagedFile, ...],
    ) -> None:
        """Make `files` and the run dir's `input/` readable in the container."""
        ...

    def collect_outputs(self, handle: ContainerHandle, *, run_dir: Path) -> None:
        """Ensure `run_dir/out/` holds the trajectory + prediction after the run."""
        ...


@runtime_checkable
class TraceSink(Protocol):
    """Where live trace records go. The container half pushes to this.

    `FileTraceSink` writes the canonical single-record `trajectory.jsonl`
    (behavior-preserving under a bind mount). Cloud sinks (`HttpTraceSink`,
    queues) accept the same records without a shared filesystem.
    """

    def emit(self, record: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


# A frozen, JSON-friendly view of the env passed into the container. Kept as a
# named alias so suites and backends agree on the shape without importing each
# other.
ContainerEnv = Mapping[str, str]


@dataclass(frozen=True)
class RunRequest:
    """Everything `run_suite_instance` needs for one instance, suite-resolved.

    The runner builds this from a `Suite` + the loaded instance; backends and
    transports consume it without knowing which suite produced it.
    """

    suite_name: str
    instance_id: str
    plan: ContainerPlan
    command: tuple[str, ...]
    env: ContainerEnv = field(default_factory=dict)
    extra_inputs: tuple[StagedFile, ...] = ()
