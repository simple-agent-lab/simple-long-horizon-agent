"""Local Docker backend (docker-py). Container path, behind `ContainerBackend`.

Turns a `RunSpec` into a container command (the generic in-container runner,
shipped in the wheel). `docker` is optional (the ``swebench`` extra) and imported
lazily, so the framework, `FakeBackend`, `LocalProcessBackend`, and unit tests
import this module without Docker installed.

Lifecycle is factored into `submit` (create + start, detached) and `poll` (is it
done yet → collect logs + remove), with the blocking `run` = submit + wait +
poll. Because the container is detached, the submitting host process can exit
after `submit` and a later process can `poll` the returned `RunHandle` — that is
what powers host-reentrant batches (`submit_dataset` / `reconcile_dataset`). The
container reads inputs and writes outputs/trace through the run's `ArtifactStore`,
so this backend never copies files itself.
"""

from __future__ import annotations

from typing import Any

from ..protocols import (
    ArtifactStore,
    ContainerBinding,
    RunHandle,
    RunOutcome,
    RunSpec,
)
from ..runner import build_command

BACKEND_KIND = "local-docker"


class LocalDockerBackend:
    """Run a spec as a container on the local (or DOCKER_HOST) daemon."""

    def __init__(self, *, user: str = "root", keep_container: bool = False) -> None:
        self.user = user
        self.keep_container = keep_container

    def _client(self) -> Any:
        import docker  # ty: ignore[unresolved-import]  # lazy: optional ``swebench`` extra

        return docker.from_env()

    def submit(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunHandle:
        """Create + start a detached container; return a handle without waiting."""

        del store  # the container reaches the store via `binding` (mounts/env)
        client = self._client()
        create_kwargs: dict[str, Any] = {
            "image": spec.plan.image,
            "name": spec.run_name,
            "user": self.user,
            "detach": True,
            "command": list(build_command(spec)),
            "environment": {**dict(spec.provider_env), **binding.env},
            "volumes": {k: dict(v) for k, v in binding.mounts.items()},
            "cap_add": list(spec.plan.cap_add),
        }
        if binding.add_hosts:
            create_kwargs["extra_hosts"] = dict(binding.add_hosts)
        if spec.plan.entrypoint is not None:
            create_kwargs["entrypoint"] = spec.plan.entrypoint
        if spec.plan.platform:
            create_kwargs["platform"] = spec.plan.platform
        if spec.plan.network_mode:
            create_kwargs["network_mode"] = spec.plan.network_mode

        container = client.containers.create(**create_kwargs)
        container.start()
        return RunHandle(backend_kind=BACKEND_KIND, ref=spec.run_name, run_dir="")

    def poll(self, handle: RunHandle) -> RunOutcome | None:
        """Return the outcome if the container has exited, else None.

        Reloads the container by name, so a *new* host process can poll a handle
        the original process submitted. On completion it collects logs and (unless
        `keep_container`) removes the container.
        """

        import docker  # ty: ignore[unresolved-import]
        import docker.errors  # ty: ignore[unresolved-import]

        client = self._client()
        try:
            container = client.containers.get(handle.ref)
        except docker.errors.NotFound:
            # Already removed (e.g. polled+collected before) — nothing to report.
            return None
        container.reload()
        if container.status not in ("exited", "dead"):
            return None
        state = container.attrs.get("State", {})
        status = int(state.get("ExitCode", 1))
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
        """Blocking lifecycle: submit, wait for exit, collect."""

        import docker.errors  # ty: ignore[unresolved-import]

        handle = self.submit(spec, store=store, binding=binding)
        client = self._client()
        try:
            container = client.containers.get(handle.ref)
            container.wait()
        except docker.errors.NotFound:
            pass
        outcome = self.poll(handle)
        # poll() returns None only if the container vanished; treat as failure.
        return outcome or RunOutcome(status_code=1, logs="container disappeared")
