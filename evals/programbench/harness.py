"""Host-side ProgramBench helpers shared by the suite, the runner, and scoring.

The Docker-free building blocks the ProgramBench integration needs on the host:
loading instances from the installed ``programbench`` package, resolving the
per-instance Docker image, dropping private fields before the agent sees a
record, and the provider environment / dotenv / offline-wheelhouse plumbing.

ProgramBench ships its task set *inside the wheel* (``programbench`` package's
``data/tasks/<instance_id>/``), so loading is ``load_all_instances`` rather than
a HuggingFace download (HF is only used by the official scorer to fetch test
blobs). ``programbench`` is an optional, heavy dependency (the ``programbench``
extra), imported lazily so this module imports without it.

The generic offline-wheelhouse builders are reused from the SWE-bench host
helpers (``evals/swebench/harness.py``) — they are benchmark-agnostic wheel
plumbing, not SWE-bench-specific — to avoid duplicating ~100 lines.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any

# Reuse the generic (benchmark-agnostic) dotenv + wheelhouse helpers.
from evals.swebench.harness import (
    load_dotenv,
    prepare_wheelhouse,
    prepare_wheelhouse_for_run,
    resolve_api_kind,
)
import simple_long_horizon_agent.config as config
from simple_long_horizon_agent.agent_flavors import (
    AGENT_FLAVOR_ENV,
    DEFAULT_AGENT_FLAVOR,
)

# Provider / reasoning env-var names are owned by `simple_long_horizon_agent.llm.env`.
# This host-side harness only forwards them into the container.
from simple_long_horizon_agent.llm.env import (
    API_KIND_ENV,
    OPENAI_API_KIND_CHOICES,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
    container_provider_env,
)

ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "API_KIND_CHOICES",
    "API_KIND_ENV",
    "AGENT_FLAVOR_ENV",
    "DEFAULT_AGENT_FLAVOR",
    "DEFAULT_IMAGE_TAG",
    "DEFAULT_RUN_ROOT",
    "DEFAULT_SCORE_IMAGE_TAG",
    "DEFAULT_UV_BINARY",
    "DEFAULT_WHEELHOUSE",
    "DEFAULT_WHEELHOUSE_MOUNT",
    "DEFAULT_WORKDIR",
    "OPENAI_AUTH_ENV",
    "OPENAI_MODEL_ENV",
    "PRIVATE_INSTANCE_FIELDS",
    "container_environment",
    "image_for_instance",
    "load_dotenv",
    "load_instance",
    "load_instances",
    "prepare_wheelhouse",
    "prepare_wheelhouse_for_run",
    "resolve_api_kind",
    "sanitized_instance",
]

# ProgramBench's cleanroom image (no build artifacts) is what the agent runs in;
# the official scorer rebuilds in the ``task`` image.
DEFAULT_IMAGE_TAG = "task_cleanroom"
DEFAULT_SCORE_IMAGE_TAG = "task"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_RUN_ROOT = ROOT / "evals/out/programbench"
DEFAULT_WHEELHOUSE = ROOT / "evals/out/programbench/wheelhouse/cp311-manylinux"
DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"
DEFAULT_UV_BINARY = shutil.which("uv") or ""

API_KIND_CHOICES = OPENAI_API_KIND_CHOICES
OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
    API_KIND_ENV,
    REASONING_EFFORT_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    AGENT_FLAVOR_ENV,
    config.COMPRESSION_THRESHOLD.name,
    config.COMPRESSION_WINDOW_RATIO.name,
    config.COMPRESSION_KEEP_RECENT.name,
)

# Gold / project-identity fields kept out of the agent-visible instance. The
# reverse-engineering task signal is the workspace's ``./executable`` + docs, so
# the agent needs none of these; dropping repo/commit also limits identity leaks
# that the rules forbid acting on (and the bash tool is network-isolated anyway).
PRIVATE_INSTANCE_FIELDS = {
    "repository",
    "commit",
    "image_name",
    "eval_clean_hashes",
    "branches",
    "tests",
    "ignored_tests",
    "ignored_branches",
}


# --------------------------------------------------------------------------- #
# Instance loading (from the installed programbench package)
# --------------------------------------------------------------------------- #
def load_instances(
    *,
    instance_ids: list[str] | None = None,
    filter_spec: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
    include_tests: bool = False,
) -> list[dict[str, Any]]:
    """Load ProgramBench instances from the installed ``programbench`` package.

    Inference does not need the gold tests, so ``include_tests`` defaults to
    False (matching mini-swe-agent's runner). ``instance_ids`` selects an exact
    subset; ``filter_spec`` / ``slice_spec`` / ``shuffle`` mirror the official
    ``filter_instances`` knobs.
    """

    load_data = importlib.import_module("programbench.utils.load_data")
    instance_filters = importlib.import_module("programbench.utils.instance_filters")
    instances = [
        dict(record)
        for record in load_data.load_all_instances(include_tests=include_tests)
    ]
    if instance_ids:
        wanted = set(instance_ids)
        instances = [i for i in instances if str(i.get("instance_id")) in wanted]
    return instance_filters.filter_instances(
        instances,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
    )


def load_instance(instance_id: str) -> dict[str, Any]:
    """Load a single ProgramBench instance by id."""

    matches = load_instances(instance_ids=[instance_id])
    if not matches:
        raise SystemExit(
            f"Instance {instance_id!r} not found in the installed programbench "
            "task set (is the 'programbench' extra installed?)."
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# Image resolution + instance shaping
# --------------------------------------------------------------------------- #
def image_for_instance(
    instance: dict[str, Any], *, image_tag: str = DEFAULT_IMAGE_TAG
) -> str:
    """Return the Docker image key for a ProgramBench instance.

    Uses the ``image_name`` the loader injects (``programbench/<org>_1776_<repo>
    .<commit>``); falls back to deriving it from ``instance_id`` so a record
    without ``image_name`` (e.g. a hand-written test fixture) still resolves.
    """

    name = str(instance.get("image_name") or "").strip()
    if not name:
        constants = importlib.import_module("programbench.constants")
        name = constants.image_name_from_instance_id(str(instance["instance_id"]))
    return f"{name}:{image_tag}"


def sanitized_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Drop gold / project-identity fields before the agent sees the record."""

    return {
        str(key): value
        for key, value in instance.items()
        if str(key) not in PRIVATE_INSTANCE_FIELDS
    }


# --------------------------------------------------------------------------- #
# Provider environment
# --------------------------------------------------------------------------- #
def container_environment(provider: str) -> dict[str, str]:
    """Collect the provider env vars passed into the container (openai only)."""

    return container_provider_env(provider, OPENAI_PASSTHROUGH_ENVS)
