"""Generic containerized eval framework (ADR 0017).

Add a benchmark suite by implementing a small `Suite` (host half: image +
launch shape, instance sanitization, prediction shape) and a container module
(``build_task`` + ``extract_result``). The framework supplies the container
lifecycle, the Python/uv bootstrap, the run-directory convention, and the one
artifact seam — parameterized over two swappable axes so the same suite runs
locally or against a cloud backend:

- `ContainerBackend` (*where compute runs*): `LocalDockerBackend` (today) →
  remote / k8s later.
- `ArtifactStore` (*where bytes live*): `LocalDirStore` (bind mount, today) →
  `HostHttpStore` (batteries-included, no third-party middleware) → `S3Store`.

Inputs, the result, and the live trajectory all flow through the one
`ArtifactStore`; there is no separate transport or trace sink. Drive one
instance with `run_suite_instance(...)`. See `evals/swebench/suite.py` (host
half) + `simple_agent_lab.evals.suites.swebench.container` (the two functions)
and ``evals/README.md`` for the "add a suite" steps.
"""

from __future__ import annotations

from .backends import (
    FakeBackend,
    LocalDockerBackend,
    LocalProcessBackend,
    RemoteDockerBackend,
)
from .bootstrap import bootstrap_script
from .protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    AgentSpec,
    ArtifactStore,
    ContainerBackend,
    ContainerBinding,
    ContainerPlan,
    ContainerTask,
    RunArtifacts,
    RunOutcome,
    RunSpec,
    Suite,
)
from .dataset import DatasetReport, InstanceResult, run_dataset
from .runner import (
    RunPaths,
    build_command,
    container_name,
    prepare_run_directory,
    run_suite_instance,
)
from .stores import (
    HostHttpStore,
    HttpArtifactClient,
    LocalDirStore,
    S3Store,
    container_store_from_env,
)

__all__ = [
    "INSTANCE_KEY",
    "RESULT_KEY",
    "TRACE_KEY",
    "AgentSpec",
    "ArtifactStore",
    "ContainerBackend",
    "ContainerBinding",
    "ContainerPlan",
    "ContainerTask",
    "DatasetReport",
    "FakeBackend",
    "InstanceResult",
    "HostHttpStore",
    "HttpArtifactClient",
    "LocalDirStore",
    "LocalDockerBackend",
    "LocalProcessBackend",
    "RemoteDockerBackend",
    "RunArtifacts",
    "RunOutcome",
    "RunPaths",
    "RunSpec",
    "S3Store",
    "Suite",
    "bootstrap_script",
    "build_command",
    "container_name",
    "container_store_from_env",
    "prepare_run_directory",
    "run_dataset",
    "run_suite_instance",
]

# `in_container` (the generic runner) pulls in the agent runtime; import it
# lazily via ``simple_agent_lab.evals.in_container`` so host-only callers
# (backends, stores, run_suite_instance) stay lightweight.
