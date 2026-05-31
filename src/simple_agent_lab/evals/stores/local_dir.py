"""Local-directory artifact store: the bind-mount default.

The per-instance run directory is bind-mounted into the container, so host and
container `get`/`put` the same files with zero copy and the lowest-latency live
trace. Assumes the Docker daemon shares this filesystem; for a remote daemon use
`HostHttpStore` (no third-party middleware) or `S3Store`.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..protocols import ContainerBinding

DEFAULT_CONTAINER_MOUNT = "/agent/run"


class LocalDirStore:
    """Keyed bytes backed by a directory. Same class works host- and container-side."""

    def __init__(
        self, root: str | Path, *, container_mount: str = DEFAULT_CONTAINER_MOUNT
    ):
        self.root = Path(root)
        self.container_mount = container_mount

    def bind(self, run_dir: Path) -> "LocalDirStore":
        return LocalDirStore(run_dir, container_mount=self.container_mount)

    def _path(self, key: str) -> Path:
        return self.root / key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a viewer tailing the trajectory never sees a torn file.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def container_binding(self) -> ContainerBinding:
        return ContainerBinding(
            mounts={
                str(self.root.resolve()): {"bind": self.container_mount, "mode": "rw"}
            },
            env={"SAL_STORE": "localdir", "SAL_STORE_ROOT": self.container_mount},
        )

    def collect_outputs(self) -> None:
        # Bind mount: the container already wrote outputs onto the host disk.
        return None
