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

import hashlib
import importlib
import os
import shutil
import sys
import tarfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the generic (benchmark-agnostic) dotenv + wheelhouse helpers.
from evals.swebench.harness import (  # noqa: E402
    load_dotenv,
    prepare_wheelhouse,
    prepare_wheelhouse_for_run,
    resolve_api_kind,
)

__all__ = [
    "API_KIND_CHOICES",
    "API_KIND_ENV",
    "DEFAULT_AGENT_FLAVOR",
    "DEFAULT_IMAGE_TAG",
    "DEFAULT_NODE_BINARY",
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
    "prepare_node_runtime",
    "resolve_api_kind",
    "sanitized_instance",
]

# ProgramBench's cleanroom image (no build artifacts) is what the agent runs in;
# the official scorer rebuilds in the ``task`` image. See
# `programbench-reverse-engineering-adapter`.
DEFAULT_IMAGE_TAG = "task_cleanroom"
DEFAULT_SCORE_IMAGE_TAG = "task"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_RUN_ROOT = ROOT / "evals/out/programbench"
DEFAULT_WHEELHOUSE = ROOT / "evals/out/programbench/wheelhouse/cp311-manylinux"
DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"
DEFAULT_UV_BINARY = shutil.which("uv") or ""

# ProgramBench task images do not ship Node, while DynamicWorkflowRuntime uses
# it as the restricted JavaScript engine. Keep a pinned Linux x64 binary inside
# the already-mounted wheelhouse so no generic Docker mount seam is needed.
NODE_VERSION = "v24.18.0"
NODE_DIST_NAME = f"node-{NODE_VERSION}-linux-x64"
NODE_ARCHIVE_NAME = f"{NODE_DIST_NAME}.tar.xz"
NODE_DOWNLOAD_URL = f"https://nodejs.org/dist/{NODE_VERSION}/{NODE_ARCHIVE_NAME}"
NODE_ARCHIVE_SHA256 = "55aa7153f9d88f28d765fcdad5ae6945b5c0f98a36881703817e4c450fa76742"
NODE_BINARY_SHA256 = "41a74efb34cbde5c7632cdac0cf8bd1a14d0b8d73dc1e82755014d9a9ce70f5c"
DEFAULT_NODE_BINARY = f"{DEFAULT_WHEELHOUSE_MOUNT}/{NODE_DIST_NAME}/bin/node"

OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
# Reasoning depth knob read by the in-container provider. Without forwarding
# these, the agent silently runs at the endpoint's default (no/low reasoning)
# even when the operator set OPENAI_REASONING_EFFORT=high in .env.
REASONING_EFFORT_ENV = "REASONING_EFFORT"
OPENAI_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
DEFAULT_AGENT_FLAVOR = "bash"
OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
    API_KIND_ENV,
    REASONING_EFFORT_ENV,
    OPENAI_REASONING_EFFORT_ENV,
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


def prepare_node_runtime(
    wheelhouse: Path,
    *,
    url: str = NODE_DOWNLOAD_URL,
    sha256: str = NODE_ARCHIVE_SHA256,
    binary_sha256: str = NODE_BINARY_SHA256,
    downloader: Callable[[str, Path], None] | None = None,
) -> Path:
    """Cache the pinned Linux Node binary under the mounted wheelhouse.

    Only ``bin/node`` is extracted: the workflow runtime does not need npm or
    Node headers. The archive checksum is verified before extraction, and a
    partial download is never promoted into the cache.
    """

    wheelhouse = Path(wheelhouse)
    _ensure_node_cache_directories(wheelhouse)
    target = wheelhouse / NODE_DIST_NAME / "bin" / "node"
    if _valid_cached_file(target, binary_sha256):
        target.chmod(0o755)
        return target
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        raise RuntimeError(f"Node cache target is not a regular file: {target}")

    archive = wheelhouse / NODE_ARCHIVE_NAME
    if not _valid_cached_file(archive, sha256):
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        partial = wheelhouse / f".{NODE_ARCHIVE_NAME}.{token}.part"
        fetch = downloader or _download_file
        try:
            fetch(url, partial)
            actual = _sha256_file(partial)
            if actual != sha256:
                raise RuntimeError(
                    f"Node archive checksum mismatch: expected {sha256}, got {actual}"
                )
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)

    # Another process may have populated the binary while this process fetched
    # the archive. Reuse it only after verifying the pinned binary digest.
    if _valid_cached_file(target, binary_sha256):
        target.chmod(0o755)
        return target

    member_name = f"{NODE_DIST_NAME}/bin/node"
    with tarfile.open(archive, mode="r:xz") as bundle:
        try:
            member = bundle.getmember(member_name)
        except KeyError as exc:
            raise RuntimeError(f"Node archive is missing {member_name}") from exc
        if not member.isfile():
            raise RuntimeError(f"Node archive member is not a file: {member_name}")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read Node archive member: {member_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        partial_target = target.with_name(f".node.{token}.part")
        try:
            with partial_target.open("wb") as output:
                shutil.copyfileobj(source, output)
            actual_binary = _sha256_file(partial_target)
            if actual_binary != binary_sha256:
                raise RuntimeError(
                    "Node binary checksum mismatch: "
                    f"expected {binary_sha256}, got {actual_binary}"
                )
            partial_target.chmod(0o755)
            partial_target.replace(target)
        finally:
            partial_target.unlink(missing_ok=True)
    target.chmod(0o755)
    return target


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cached_file(path: Path, sha256: str) -> bool:
    return not path.is_symlink() and path.is_file() and _sha256_file(path) == sha256


def _ensure_node_cache_directories(wheelhouse: Path) -> None:
    if wheelhouse.is_symlink():
        raise RuntimeError(f"Node wheelhouse must not be a symlink: {wheelhouse}")
    wheelhouse.mkdir(parents=True, exist_ok=True)
    for directory in (
        wheelhouse / NODE_DIST_NAME,
        wheelhouse / NODE_DIST_NAME / "bin",
    ):
        if directory.is_symlink():
            raise RuntimeError(
                f"Node cache directory must not be a symlink: {directory}"
            )
        if directory.exists() and not directory.is_dir():
            raise RuntimeError(f"Node cache path is not a directory: {directory}")
        directory.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Provider environment
# --------------------------------------------------------------------------- #
def container_environment(provider: str) -> dict[str, str]:
    """Collect the provider env vars passed into the container (openai only)."""

    env: dict[str, str] = {}
    if provider != "openai":
        return env
    for name in OPENAI_PASSTHROUGH_ENVS:
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    missing = [name for name in (OPENAI_MODEL_ENV, OPENAI_AUTH_ENV) if name not in env]
    if missing:
        raise SystemExit(
            "Missing required env vars for --provider openai: " + ", ".join(missing)
        )
    return env
