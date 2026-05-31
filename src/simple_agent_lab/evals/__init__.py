"""Generic containerized eval framework (ADR 0017).

Add a benchmark suite by implementing a small `Suite` (host half: image +
launch shape, instance sanitization, prediction shape) and a container module
(``build_task`` + ``extract_result``). The framework supplies the container
lifecycle, the Python/uv bootstrap, the run-directory convention, and artifact
transport — parameterized over three swappable seams so the same suite runs
locally or against a cloud backend:

- `ContainerBackend`: `LocalDockerBackend` (today) → remote/cloud later.
- `ArtifactTransport`: `BindMountTransport` (today) → `CopyOutTransport` (cloud).
- `TraceSink`: `FileTraceSink` (today) → `HttpTraceSink` (cloud).

Drive one instance with `run_suite_instance(...)`. See `evals/swebench/suite.py`
for the reference `Suite` and ``evals/README.md`` for the "add a suite" steps.
"""

from __future__ import annotations

from .backends import FakeBackend, LocalDockerBackend
from .bootstrap import bootstrap_script
from .protocols import (
    RESULT_FILE,
    AgentSpec,
    ArtifactTransport,
    ContainerBackend,
    ContainerHandle,
    ContainerPlan,
    ContainerTask,
    RunArtifacts,
    RunRequest,
    StagedFile,
    Suite,
    TraceSink,
)
from .runner import RunPaths, container_name, prepare_run_directory, run_suite_instance
from .trace_sink import FileTraceSink, HttpTraceSink
from .transport import BindMountTransport, CopyOutTransport

__all__ = [
    "RESULT_FILE",
    "AgentSpec",
    "ArtifactTransport",
    "BindMountTransport",
    "ContainerBackend",
    "ContainerHandle",
    "ContainerPlan",
    "ContainerTask",
    "CopyOutTransport",
    "FakeBackend",
    "FileTraceSink",
    "HttpTraceSink",
    "LocalDockerBackend",
    "RunArtifacts",
    "RunPaths",
    "RunRequest",
    "StagedFile",
    "Suite",
    "TraceSink",
    "bootstrap_script",
    "container_name",
    "prepare_run_directory",
    "run_suite_instance",
]

# `in_container` (the generic runner) pulls in the agent runtime; import it
# lazily via ``simple_agent_lab.evals.in_container`` so host-only callers
# (backends, transports, run_suite_instance) stay lightweight.
