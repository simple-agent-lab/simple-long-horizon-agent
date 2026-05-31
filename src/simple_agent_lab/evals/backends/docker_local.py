"""Local Docker backend (docker-py). Today's behavior, behind the protocol.

Wraps the container lifecycle (create → start → wait → logs → remove) so the
generic runner depends on `ContainerBackend`, not on docker-py. `docker` is an
optional dependency (the ``swebench`` extra), so it is imported lazily: the
framework, `FakeBackend`, and unit tests import this module without Docker
installed.

A future `RemoteDockerBackend` can reuse the same handle by pointing docker-py
at ``DOCKER_HOST``; combined with `HostHttpStore`/`S3Store` that is the cloud
path (ADR 0017). The container reads inputs and writes outputs/trace through the
run's `ArtifactStore`, so this backend never copies files itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import ContainerPlan


class LocalDockerContainerHandle:
    """A created docker-py container, exposed through `ContainerHandle`."""

    def __init__(self, container: Any) -> None:
        self._container = container

    def start(self) -> None:
        self._container.start()

    def wait(self) -> int:
        result = self._container.wait()
        return int(result.get("StatusCode", 1))

    def logs(self) -> str:
        return self._container.logs(stdout=True, stderr=True).decode(errors="replace")

    def remove(self) -> None:
        self._container.remove(force=True)


class LocalDockerBackend:
    """Create containers on the local (or DOCKER_HOST) daemon via docker-py."""

    def __init__(self, *, user: str = "root") -> None:
        self.user = user

    def create(
        self,
        *,
        name: str,
        plan: ContainerPlan,
        command: tuple[str, ...],
        env: Mapping[str, str],
        mounts: Mapping[str, Mapping[str, str]],
        add_hosts: Mapping[str, str] | None = None,
    ) -> LocalDockerContainerHandle:
        import docker  # ty: ignore[unresolved-import]  # lazy: optional ``swebench`` extra

        client = docker.from_env()
        create_kwargs: dict[str, Any] = {
            "image": plan.image,
            "name": name,
            "user": self.user,
            "detach": True,
            "command": list(command),
            "environment": dict(env),
            "volumes": {k: dict(v) for k, v in mounts.items()},
            "cap_add": list(plan.cap_add),
        }
        if add_hosts:
            create_kwargs["extra_hosts"] = dict(add_hosts)
        if plan.entrypoint is not None:
            create_kwargs["entrypoint"] = plan.entrypoint
        if plan.platform:
            create_kwargs["platform"] = plan.platform
        if plan.network_mode:
            create_kwargs["network_mode"] = plan.network_mode
        container = client.containers.create(**create_kwargs)
        return LocalDockerContainerHandle(container)
