"""Artifact transports: how inputs reach the container and outputs return.

`BindMountTransport` is the local default (shared filesystem, zero-copy,
lowest-latency live trace). `CopyOutTransport` is the cloud path (archive
in / archive out, no shared filesystem) and is a documented stub today.
"""

from __future__ import annotations

from .bind_mount import BindMountTransport
from .copy_out import CopyOutTransport

__all__ = ["BindMountTransport", "CopyOutTransport"]
