"""Artifact stores: the one seam for inputs, outputs, and the live trajectory.

- `LocalDirStore` — bind-mount default, single machine.
- `HostHttpStore` / `HttpArtifactClient` — host runs a stdlib HTTP store; works
  across a remote daemon with no third-party middleware.
- `S3Store` — production stub for fully decoupled runs.

`container_store_from_env` reconstructs the container-side store from the env
the host's `ContainerBinding` set.
"""

from __future__ import annotations

import os

from .host_http import HostHttpStore, HttpArtifactClient
from .local_dir import LocalDirStore
from .s3 import S3Store


def container_store_from_env(env: dict[str, str] | None = None):
    """Build the container-side `ArtifactStore` the in-container runner uses."""

    source = env if env is not None else os.environ
    kind = source.get("SAL_STORE", "localdir")
    if kind == "localdir":
        return LocalDirStore(source.get("SAL_STORE_ROOT", "/agent/run"))
    if kind == "http":
        return HttpArtifactClient(
            source["SAL_STORE_URL"], source.get("SAL_STORE_TOKEN", "")
        )
    raise SystemExit(f"Unknown SAL_STORE kind {kind!r}")


__all__ = [
    "HostHttpStore",
    "HttpArtifactClient",
    "LocalDirStore",
    "S3Store",
    "container_store_from_env",
]
