"""Local Docker backend (docker-py). Container path, behind `ContainerBackend`.

Turns a `RunSpec` into a container command (the generic in-container runner,
shipped in the wheel) and owns the full lifecycle: create → start → wait → logs
→ remove. `docker` is optional (the ``swebench`` extra) and imported lazily, so
the framework, `FakeBackend`, `LocalProcessBackend`, and unit tests import this
module without Docker installed.

A future `RemoteDockerBackend` is this class pointed at ``DOCKER_HOST``; combined
with `HostHttpStore`/`S3Store` that is the multi-machine path (ADR 0017). The
container reads inputs and writes outputs/trace through the run's
`ArtifactStore`, so this backend never copies files itself.
"""

from __future__ import annotations

from typing import Any

from ..protocols import ArtifactStore, ContainerBinding, RunOutcome, RunSpec
from ..runner import build_command


class LocalDockerBackend:
    """Run a spec as a container on the local (or DOCKER_HOST) daemon."""

    def __init__(self, *, user: str = "root", keep_container: bool = False) -> None:
        self.user = user
        self.keep_container = keep_container

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome:
        import docker  # ty: ignore[unresolved-import]  # lazy: optional ``swebench`` extra

        del store  # the container reaches the store via `binding` (mounts/env)
        client = docker.from_env()
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
        try:
            container.start()
            result = container.wait()
            status = int(result.get("StatusCode", 1))
            logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
        finally:
            if not self.keep_container:
                container.remove(force=True)
        return RunOutcome(status_code=status, logs=logs)
