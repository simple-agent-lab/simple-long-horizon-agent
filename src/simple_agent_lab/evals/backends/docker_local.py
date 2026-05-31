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
    """Run a spec as a container on the local (or DOCKER_HOST) daemon.

    `pull`: image pull policy, like the legacy launcher — ``"missing"`` (default,
    pull only when absent), ``"always"``, or ``"never"``. docker-py's ``create``
    does not auto-pull (unlike ``run``), so without this a missing image would
    raise ``ImageNotFound`` on first use.
    """

    def __init__(
        self,
        *,
        user: str = "root",
        keep_container: bool = False,
        pull: str = "missing",
    ) -> None:
        self.user = user
        self.keep_container = keep_container
        self.pull = pull

    def _client(self) -> Any:
        import docker  # ty: ignore[unresolved-import]  # lazy: optional ``swebench`` extra

        return docker.from_env()

    def _ensure_image(self, client: Any, image: str, platform: str | None) -> None:
        """Apply the pull policy before create() (which never auto-pulls)."""

        import docker.errors  # ty: ignore[unresolved-import]

        if self.pull == "always":
            client.images.pull(image, platform=platform)
            return
        try:
            client.images.get(image)
        except docker.errors.ImageNotFound:
            if self.pull == "never":
                raise
            client.images.pull(image, platform=platform)

    def submit(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunHandle:
        """Pull (per policy) + create + start a detached container; no wait."""

        import docker.errors  # ty: ignore[unresolved-import]

        del store  # the container reaches the store via `binding` (mounts/env)
        client = self._client()
        self._ensure_image(client, spec.plan.image, spec.plan.platform or None)
        create_kwargs = _create_kwargs(
            spec,
            binding,
            user=self.user,
            environment={**dict(spec.provider_env), **binding.env},
        )
        container = client.containers.create(**create_kwargs)
        try:
            container.start()
        except docker.errors.APIError:
            # Don't leave the named container behind, or the next attempt (retry /
            # resubmit) collides on the deterministic name with a 409.
            container.remove(force=True)
            raise
        return RunHandle(backend_kind=BACKEND_KIND, ref=spec.run_name, run_dir="")

    def poll(self, handle: RunHandle) -> RunOutcome | None:
        """Return the outcome if the container has exited, else None.

        Reloads the container by name, so a *new* host process can poll a handle
        the original process submitted. On completion it collects logs and (unless
        `keep_container`) removes the container.
        """

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
        # ExitCode can be null on odd terminal states; treat as failure, not crash.
        raw_code = state.get("ExitCode")
        status = int(raw_code) if isinstance(raw_code, int) else 1
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

        import docker.errors  # ty: ignore[unresolved-import]

        handle = self.submit(spec, store=store, binding=binding)
        client = self._client()
        try:
            client.containers.get(handle.ref).wait()
        except docker.errors.NotFound:
            pass  # finished + reaped already; poll() reads the terminal state
        outcome = self.poll(handle)
        # poll() returns None only if the container vanished after wait(); the run
        # itself wrote result.json, so report success — collect_outputs/_shape_
        # prediction downstream still find the artifacts.
        return outcome or RunOutcome(status_code=0, logs="")


def _create_kwargs(
    spec: RunSpec,
    binding: ContainerBinding,
    *,
    user: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Assemble docker-py create() kwargs from a plan + binding (shared by backends)."""

    kwargs: dict[str, Any] = {
        "image": spec.plan.image,
        "name": spec.run_name,
        "user": user,
        "detach": True,
        "command": list(build_command(spec)),
        "environment": environment,
        "volumes": {k: dict(v) for k, v in binding.mounts.items()},
        "cap_add": list(spec.plan.cap_add),
    }
    if binding.add_hosts:
        kwargs["extra_hosts"] = dict(binding.add_hosts)
    if spec.plan.entrypoint is not None:
        kwargs["entrypoint"] = spec.plan.entrypoint
    if spec.plan.platform:
        kwargs["platform"] = spec.plan.platform
    if spec.plan.network_mode:
        kwargs["network_mode"] = spec.plan.network_mode
    return kwargs
