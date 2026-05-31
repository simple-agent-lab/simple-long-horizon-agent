"""Remote Docker backend: host-pull, for the common "host → worker only" topology.

Reality check: a host can usually reach its workers (it drives their daemon over
``DOCKER_HOST=tcp://…`` or SSH), but a worker behind NAT often *cannot* reach
back to the host. `HostHttpStore` assumes the reverse (worker → host), so it does
not fit here.

This backend reverses the data flow to **host-pull**, reusing the one connection
that already exists — the host→worker docker API:

  - the worker container writes only to its *own* filesystem (`SAL_STORE=localdir`
    at an in-container path) — zero network, zero reverse connection;
  - the host pushes the instance in with ``put_archive`` before start, and pulls
    ``out/`` back with ``get_archive`` after (and, optionally, polls it during
    the run for a live trace) — all over the outbound host→worker connection.

So the worker needs no inbound reachability at all. The tar (un)packing lives in
``_archive`` as pure functions; this module is the docker orchestration around
them, mirroring `LocalDockerBackend` but with no bind mount.
"""

from __future__ import annotations

import threading
from typing import Any

from ..protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    ArtifactStore,
    ContainerBinding,
    RunOutcome,
    RunSpec,
)
from ..runner import build_command
from ._archive import pack_file_to_root, read_stream, unpack_members

# Where the worker container keeps its run dir. No host path is shared, so this
# is purely in-container; the host moves bytes in/out by tar over the daemon API.
RUN_MOUNT = "/agent/run"


def push_inputs(
    container: Any, store: ArtifactStore, *, run_mount: str = RUN_MOUNT
) -> None:
    """Copy the seeded inputs from the host store into the container (host→worker)."""

    data = store.get(INSTANCE_KEY)
    tar = pack_file_to_root(f"{run_mount}/{INSTANCE_KEY}", data)
    container.put_archive("/", tar)


def pull_outputs(
    container: Any,
    store: ArtifactStore,
    *,
    run_mount: str = RUN_MOUNT,
    keys: tuple[str, ...] = (RESULT_KEY, TRACE_KEY),
) -> list[str]:
    """Copy ``out/`` artifacts from the container back into the host store.

    Returns the keys actually pulled. ``get_archive`` of the ``out`` dir yields
    members named ``out/<file>``, which already match the store keys.
    """

    try:
        stream, _stat = container.get_archive(f"{run_mount}/out")
    except Exception:
        return []
    members = unpack_members(read_stream(stream))
    wanted = set(keys)
    pulled: list[str] = []
    for name, data in members.items():
        if name in wanted:
            store.put(name, data)
            pulled.append(name)
    return pulled


class RemoteDockerBackend:
    """Run a spec on a remote daemon, moving artifacts by host-initiated tar copy."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        user: str = "root",
        keep_container: bool = False,
        live_poll_interval_s: float = 0.0,
    ) -> None:
        self.base_url = base_url
        self.user = user
        self.keep_container = keep_container
        # >0 enables a background thread that pulls out/trajectory.jsonl on this
        # cadence so the local trace viewer can tail it during the run.
        self.live_poll_interval_s = live_poll_interval_s

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome:
        import docker  # ty: ignore[unresolved-import]  # lazy: optional ``swebench`` extra

        del binding  # remote ignores host mounts/env; the worker uses a local store
        client = (
            docker.DockerClient(base_url=self.base_url)
            if self.base_url
            else docker.from_env()
        )
        env = {
            **dict(spec.provider_env),
            "SAL_STORE": "localdir",
            "SAL_STORE_ROOT": RUN_MOUNT,
        }
        create_kwargs: dict[str, Any] = {
            "image": spec.plan.image,
            "name": spec.run_name,
            "user": self.user,
            "detach": True,
            "command": list(build_command(spec)),
            "environment": env,
            "cap_add": list(spec.plan.cap_add),
        }
        if spec.plan.entrypoint is not None:
            create_kwargs["entrypoint"] = spec.plan.entrypoint
        if spec.plan.platform:
            create_kwargs["platform"] = spec.plan.platform
        if spec.plan.network_mode:
            create_kwargs["network_mode"] = spec.plan.network_mode

        container = client.containers.create(**create_kwargs)
        stop = threading.Event()
        poller: threading.Thread | None = None
        try:
            push_inputs(container, store)
            container.start()
            if self.live_poll_interval_s > 0:
                poller = self._start_live_poller(container, store, stop)
            result = container.wait()
            status = int(result.get("StatusCode", 1))
            logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
        finally:
            stop.set()
            if poller is not None:
                poller.join(timeout=self.live_poll_interval_s + 5)
            pulled = pull_outputs(container, store)
            if not self.keep_container:
                container.remove(force=True)
        del pulled
        return RunOutcome(status_code=status, logs=logs)

    def _start_live_poller(
        self, container: Any, store: ArtifactStore, stop: threading.Event
    ) -> threading.Thread:
        def loop() -> None:
            while not stop.wait(self.live_poll_interval_s):
                pull_outputs(container, store, keys=(TRACE_KEY,))

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return thread
