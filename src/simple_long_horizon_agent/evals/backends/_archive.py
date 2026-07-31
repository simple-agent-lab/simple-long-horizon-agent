"""Tar (un)packing for Docker's ``put_archive`` / ``get_archive`` APIs.

Pure functions, no docker dependency, so the host-pull copy logic of
`RemoteDockerBackend` is unit-testable without a daemon. ``put_archive`` extracts
a tar into a target dir; ``get_archive`` returns a tar of a path. These helpers
build the outbound tar (with parent dir entries, so it can be written to ``/``
even when intermediate dirs do not exist) and read members back out.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable
from pathlib import PurePosixPath


def pack_file_to_root(container_path: str, data: bytes, *, mode: int = 0o644) -> bytes:
    """Tar one file at an absolute container path, with its parent dirs.

    Written to be applied with ``put_archive("/", ...)`` so missing intermediate
    directories are created by the extraction.
    """

    arc = PurePosixPath(container_path.lstrip("/"))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        current = PurePosixPath()
        for part in arc.parent.parts:
            current = current / part
            info = tarfile.TarInfo(str(current))
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        info = tarfile.TarInfo(str(arc))
        info.size = len(data)
        info.mode = mode
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def unpack_members(tar_bytes: bytes) -> dict[str, bytes]:
    """Return ``{posix_relpath: data}`` for every regular file in a tar stream.

    Keys are the archive member names as docker emits them: ``get_archive`` of a
    file yields a single member named by basename; of a directory it yields
    ``<dirname>/...`` members.
    """

    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is not None:
                out[member.name] = extracted.read()
    return out


def read_stream(stream: Iterable[bytes]) -> bytes:
    """Join docker's ``get_archive`` chunk iterator into one bytes blob."""

    return b"".join(stream)
