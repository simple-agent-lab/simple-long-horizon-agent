"""Local Docker backend (docker-py). Today's behavior, behind the protocol.

This wraps the container lifecycle that `evals/swebench/containerized_agent.py`
performs inline (create → put_archive → start → wait → logs → remove) so the
generic runner depends on `ContainerBackend`, not on docker-py. `docker` is an
optional dependency (the ``swebench`` extra), so it is imported lazily: the
framework, `FakeBackend`, and unit tests import this module without Docker
installed.

A future `RemoteDockerBackend` can reuse the same handle by pointing docker-py
at ``DOCKER_HOST``; combined with `CopyOutTransport` that is the cloud path
(ADR 0017).
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..protocols import ContainerPlan, StagedFile


def _put_archive(container: Any, *, data: bytes, target_path: str, mode: int) -> None:
    """Copy one file into a container using Docker's tar archive API."""

    archive_path = target_path.lstrip("/")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        current = Path()
        for part in Path(archive_path).parent.parts:
            current = current / part
            info = tarfile.TarInfo(current.as_posix())
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        info = tarfile.TarInfo(archive_path)
        info.size = len(data)
        info.mode = mode
        archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    container.put_archive("/", buffer.getvalue())


class LocalDockerContainerHandle:
    """A created docker-py container, exposed through `ContainerHandle`."""

    def __init__(self, container: Any) -> None:
        self._container = container

    def put_file(self, file: StagedFile) -> None:
        _put_archive(
            self._container,
            data=file.data,
            target_path=file.container_path,
            mode=file.mode,
        )

    def start(self) -> None:
        self._container.start()

    def wait(self) -> int:
        result = self._container.wait()
        return int(result.get("StatusCode", 1))

    def logs(self) -> str:
        return self._container.logs(stdout=True, stderr=True).decode(errors="replace")

    def get_archive(self, container_path: str) -> bytes:
        stream, _stat = self._container.get_archive(container_path)
        return b"".join(stream)

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
        if plan.entrypoint is not None:
            create_kwargs["entrypoint"] = plan.entrypoint
        if plan.platform:
            create_kwargs["platform"] = plan.platform
        if plan.network_mode:
            create_kwargs["network_mode"] = plan.network_mode
        container = client.containers.create(**create_kwargs)
        return LocalDockerContainerHandle(container)
