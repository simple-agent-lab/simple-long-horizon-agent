"""S3 artifact store: the production path. Documented stub (ADR 0017).

Unlike `LocalDirStore` (shared filesystem) and `HostHttpStore` (host serves
the bytes), an object store decouples the run from the host entirely: the
container uploads inputs/outputs/trace to a bucket, and the host downloads them
later. That is what lets the host go offline between submit and result-fetch.

Implementation will `put`/`get` against a bucket+prefix and surface credentials
to the container via `container_binding().env`. Kept as a stub so the seam is
fixed; suites and the runner never change when it lands.
"""

from __future__ import annotations

from pathlib import Path

from ..protocols import ContainerBinding


class S3Store:
    """Object-store artifact transport for fully decoupled runs. Not implemented."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix

    def bind(self, run_dir: Path) -> "S3Store":  # pragma: no cover - stub
        return S3Store(self.bucket, f"{self.prefix}/{Path(run_dir).name}")

    def get(self, key: str) -> bytes:  # pragma: no cover - stub
        raise NotImplementedError(
            "S3Store is a documented stub (ADR 0017). Use LocalDirStore locally "
            "or HostHttpStore for a remote daemon without S3."
        )

    def put(self, key: str, data: bytes) -> None:  # pragma: no cover - stub
        raise NotImplementedError("S3Store is a documented stub (ADR 0017).")

    def container_binding(self) -> ContainerBinding:  # pragma: no cover - stub
        raise NotImplementedError("S3Store is a documented stub (ADR 0017).")

    def collect_outputs(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError("S3Store is a documented stub (ADR 0017).")
