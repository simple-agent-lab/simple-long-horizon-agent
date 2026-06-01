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
  `HostHttpStore` (batteries-included, no third-party middleware) → an object
  store (S3/GCS) later.

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
from .batch import reconcile_dataset, submit_dataset
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
    RunHandle,
    RunOutcome,
    RunSpec,
    Suite,
)
from .dataset import DatasetReport, InstanceResult, run_dataset
from .runner import run_suite_instance
from .stores import HostHttpStore, LocalDirStore

# Public surface: the things you compose a run from — suites, backends, stores,
# the run/dataset entry points, and the protocol/value types you implement
# against. Internal helpers (`build_command`, `container_name`,
# `prepare_run_directory`/`RunPaths`, `bootstrap_script`, `container_store_from_env`,
# the container-side `HttpArtifactClient`) stay importable from their own modules
# but are kept out of the top-level facade so "what do I import from
# simple_agent_lab.evals" reads as the user surface, not the plumbing.
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
    "LocalDirStore",
    "LocalDockerBackend",
    "LocalProcessBackend",
    "RemoteDockerBackend",
    "RunArtifacts",
    "RunHandle",
    "RunOutcome",
    "RunSpec",
    "Suite",
    "reconcile_dataset",
    "run_dataset",
    "run_suite_instance",
    "submit_dataset",
]

# `in_container` (the generic runner) pulls in the agent runtime; import it
# lazily via ``simple_agent_lab.evals.in_container`` so host-only callers
# (backends, stores, run_suite_instance) stay lightweight.
