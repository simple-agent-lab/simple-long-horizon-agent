"""Container backends: where eval containers run.

`LocalDockerBackend` wraps docker-py (today's behavior). `FakeBackend` runs no
container at all — it drives the lifecycle in-memory so the framework, suites,
and transports can be unit-tested without Docker, mirroring the LLM layer's
``fake`` adapter.
"""

from __future__ import annotations

from .fake import FakeBackend
from .docker_local import LocalDockerBackend

__all__ = ["FakeBackend", "LocalDockerBackend"]
