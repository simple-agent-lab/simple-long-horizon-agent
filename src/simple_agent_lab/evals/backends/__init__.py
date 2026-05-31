"""Container backends: where a run executes.

- `LocalProcessBackend` — in-process, no Docker (local development).
- `LocalDockerBackend` — a container on the local/DOCKER_HOST daemon.
- `RemoteDockerBackend` — a remote daemon, host-pull artifacts (worker needs no
  reverse reachability to the host).
- `FakeBackend` — in-memory, for testing orchestration without an agent.

All consume the same `RunSpec` + bound `ArtifactStore`, so swapping the backend
moves a suite from local dev to multi-machine deployment with no code change.
"""

from __future__ import annotations

from .docker_local import LocalDockerBackend
from .fake import FakeBackend
from .local_process import LocalProcessBackend
from .remote_docker import RemoteDockerBackend

__all__ = [
    "FakeBackend",
    "LocalDockerBackend",
    "LocalProcessBackend",
    "RemoteDockerBackend",
]
