"""Local Docker backend (docker-py). Container path, behind `ContainerBackend`.

Turns a `RunSpec` into a container command (the generic in-container runner,
shipped in the wheel). `docker` is optional (the ``swebench`` extra) and imported
lazily, so the framework, `FakeBackend`, `LocalProcessBackend`, and unit tests
import this module without Docker installed.

Lifecycle is factored into `submit` (pull-if-needed + create + start, detached)
and `poll` (is it done yet → collect logs + remove), with the blocking `run` =
submit + wait + poll. Because the container is detached, the submitting host
process can exit after `submit` and a later process can `poll` the returned
`RunHandle` — that is what powers host-reentrant batches (`submit_dataset` /
`reconcile_dataset`). The container reads inputs and writes outputs/trace through
the run's `ArtifactStore`, so this backend never copies files itself.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..bootstrap import UV_CONTAINER_PATH
from ..protocols import (
    ArtifactStore,
    ContainerBinding,
    RunHandle,
    RunOutcome,
    RunSpec,
)
from ..runner import build_command

# `docker` is the optional ``swebench`` extra: import it once at module top,
# guarded so host-only callers (FakeBackend, LocalProcessBackend, the whole
# `simple_agent_lab.evals` facade) still import on a machine without it. The
# docker backends call `_require_docker()` before any daemon op, turning a bare
# ModuleNotFoundError into an actionable install hint.
# Declared `Any` (not the inferred `module | None`) so the guarded attribute
# access below — `docker.errors.NotFound`, reached only after `_require_docker()`
# — type checks cleanly; the runtime None-guard, not the type, enforces presence.
docker: Any = None
try:  # pragma: no cover - import guard
    docker = importlib.import_module("docker")
except ImportError:  # pragma: no cover - exercised only without the extra
    pass

BACKEND_KIND = "local-docker"


def _require_docker() -> Any:
    """Return the imported ``docker`` module, or raise an actionable error."""

    if docker is None:
        raise RuntimeError(
            "Docker backends need the 'swebench' extra: "
            "pip install 'simple-agent-lab[swebench]'"
        )
    return docker


def _ensure_image(client: Any, image: str, platform: str | None, pull: str) -> None:
    """Apply the pull policy before create() (which never auto-pulls)."""

    if pull == "always":
        client.images.pull(image, platform=platform)
        return
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        if pull == "never":
            raise
        client.images.pull(image, platform=platform)


def start_container(
    client: Any,
    spec: RunSpec,
    binding: ContainerBinding,
    *,
    user: str,
    pull: str,
    environment: dict[str, str],
    before_start: Callable[[Any], None] | None = None,
) -> Any:
    """Pull-per-policy + create + (before_start) + start a detached container.

    Shared by both docker backends so the pull policy, the create kwargs, and the
    remove-on-failure guard cannot drift between local and remote. docker-py's
    ``create`` does not auto-pull (unlike ``run``), and a named container left
    behind after a failed ``start`` would 409 on the next retry — both handled here.

    `before_start` runs against the created (not yet started) container — the
    remote backend uses it to ``put_archive`` inputs in before the worker boots.
    Any failure there also triggers the remove-on-failure cleanup.
    """

    _ensure_image(
        client, spec.launch_spec.image, spec.launch_spec.platform or None, pull
    )
    create_kwargs = _create_kwargs(spec, binding, user=user, environment=environment)
    container = client.containers.create(**create_kwargs)
    try:
        if before_start is not None:
            before_start(container)
        container.start()
    except (docker.errors.APIError, OSError):
        # Don't leave the named container behind, or the next attempt (retry /
        # resubmit) collides on the deterministic name with a 409.
        container.remove(force=True)
        raise
    return container


def exit_status(state: Mapping[str, Any]) -> int:
    """Read a container's exit code from its State, treating null/odd as failure."""

    raw_code = state.get("ExitCode")
    return int(raw_code) if isinstance(raw_code, int) else 1


def with_local_mounts(
    binding: ContainerBinding,
    *,
    wheelhouse: str | Path | None,
    wheelhouse_mount: str | None,
    uv_binary: str | Path | None,
) -> ContainerBinding:
    """Add read-only bind mounts for the offline wheelhouse and an optional uv.

    `LocalDockerBackend` assumes a shared filesystem (the same assumption as
    `LocalDirStore`'s bind mount), so the offline-install path is wired by
    bind-mounting the host wheelhouse at the container's ``wheelhouse_mount`` —
    the ``pip install --no-index --find-links`` directory the bootstrap uses.
    A ``uv`` binary can be mounted at ``/tmp/uv`` (the bootstrap probes there
    first) so images that ship neither ``uv`` nor a Python 3.11 can still build
    the agent venv the wheels target. Both are read-only; the container never
    writes back to them.
    """

    extra: dict[str, dict[str, str]] = {}
    if wheelhouse:
        if not wheelhouse_mount:
            raise ValueError(
                "LocalDockerBackend(wheelhouse=...) needs the run's "
                "wheelhouse_mount (the in-container --find-links path) to be set"
            )
        extra[str(Path(wheelhouse).resolve())] = {
            "bind": wheelhouse_mount,
            "mode": "ro",
        }
    if uv_binary:
        extra[str(Path(uv_binary).resolve())] = {
            "bind": UV_CONTAINER_PATH,
            "mode": "ro",
        }
    if not extra:
        return binding
    return replace(binding, mounts={**binding.mounts, **extra})


class LocalDockerBackend:
    """Run a spec as a container on the local (or DOCKER_HOST) daemon.

    `pull`: image pull policy, like the legacy launcher — ``"missing"`` (default,
    pull only when absent), ``"always"``, or ``"never"``. docker-py's ``create``
    does not auto-pull (unlike ``run``), so without this a missing image would
    raise ``ImageNotFound`` on first use.

    `wheelhouse`: host directory of wheels for an offline install. When set it is
    bind-mounted read-only at the run's ``wheelhouse_mount`` so the container's
    bootstrap installs ``simple-agent-lab`` with ``--no-index --find-links`` (no
    PyPI). `uv_binary`: host path to a Linux ``uv`` binary, bind-mounted at
    ``/tmp/uv``, for images whose own Python predates the wheels' 3.11 target.
    """

    def __init__(
        self,
        *,
        user: str = "root",
        keep_container: bool = False,
        pull: str = "missing",
        wheelhouse: str | Path | None = None,
        uv_binary: str | Path | None = None,
    ) -> None:
        self.user = user
        self.keep_container = keep_container
        self.pull = pull
        self.wheelhouse = wheelhouse
        self.uv_binary = uv_binary

    def _client(self) -> Any:
        return _require_docker().from_env()

    def submit(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunHandle:
        """Pull (per policy) + create + start a detached container; no wait."""

        del store  # the container reaches the store via `binding` (mounts/env)
        client = self._client()
        binding = with_local_mounts(
            binding,
            wheelhouse=self.wheelhouse,
            wheelhouse_mount=spec.wheelhouse_mount,
            uv_binary=self.uv_binary,
        )
        start_container(
            client,
            spec,
            binding,
            user=self.user,
            pull=self.pull,
            environment={**dict(spec.provider_env), **binding.env},
        )
        return RunHandle(backend_kind=BACKEND_KIND, ref=spec.run_name, run_dir="")

    def poll(self, handle: RunHandle) -> RunOutcome | None:
        """Return the outcome if the container has exited, else None.

        Reloads the container by name, so a *new* host process can poll a handle
        the original process submitted. On completion it collects logs and (unless
        `keep_container`) removes the container.
        """

        client = self._client()
        try:
            container = client.containers.get(handle.ref)
        except docker.errors.NotFound:
            # Already removed (e.g. polled+collected before) — nothing to report.
            return None
        container.reload()
        if container.status not in ("exited", "dead"):
            return None
        status = exit_status(container.attrs.get("State", {}))
        logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
        if not self.keep_container:
            container.remove(force=True)
        return RunOutcome(status_code=status, logs=logs)

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome:
        """Blocking lifecycle: submit, wait for exit, collect via poll."""

        handle = self.submit(spec, store=store, binding=binding)
        client = self._client()
        try:
            client.containers.get(handle.ref).wait()
        except docker.errors.NotFound:
            pass  # finished + reaped already; poll() reads the terminal state
        outcome = self.poll(handle)
        if outcome is not None:
            return outcome
        # The container vanished before poll() could read its exit code. Trust
        # the artifact, not an assumption: a written result.json means the run
        # completed; its absence means it failed (e.g. crashed before writing).
        from ..protocols import RESULT_KEY

        wrote_result = _store_has(store, RESULT_KEY)
        return RunOutcome(
            status_code=0 if wrote_result else 1,
            logs="" if wrote_result else "container disappeared without a result",
        )


def _store_has(store: ArtifactStore, key: str) -> bool:
    try:
        store.get(key)
        return True
    except (FileNotFoundError, OSError):
        return False


def _create_kwargs(
    spec: RunSpec,
    binding: ContainerBinding,
    *,
    user: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Assemble docker-py create() kwargs from a launch spec + binding (shared by backends)."""

    kwargs: dict[str, Any] = {
        "image": spec.launch_spec.image,
        "name": spec.run_name,
        "user": user,
        "detach": True,
        "command": list(build_command(spec)),
        "environment": environment,
        "volumes": {k: dict(v) for k, v in binding.mounts.items()},
        "cap_add": list(spec.launch_spec.cap_add),
    }
    if binding.add_hosts:
        kwargs["extra_hosts"] = dict(binding.add_hosts)
    if spec.launch_spec.entrypoint is not None:
        kwargs["entrypoint"] = spec.launch_spec.entrypoint
    if spec.launch_spec.platform:
        kwargs["platform"] = spec.launch_spec.platform
    if spec.launch_spec.network_mode:
        kwargs["network_mode"] = spec.launch_spec.network_mode
    return kwargs
