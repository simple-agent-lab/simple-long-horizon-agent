"""Generic orchestration for one eval instance.

`run_suite_instance(...)` wires a `Suite` + `ContainerBackend` + `ArtifactStore`
together: it resolves the launch spec, seeds the instance into the store, hands a
`RunSpec` to the backend, and collects the result. It has no
`if pro:` branches and makes no Docker calls of its own — the suite supplies the
launch spec as data, and the backend owns the run.

The *same* call runs locally or across machines by swapping the backend:
`LocalProcessBackend` (in-process, no Docker — local dev) /
`LocalDockerBackend` (one machine) / a remote backend (multi-machine). The
suite's container half runs identically either way, because every backend reads
the instance and writes the result/trajectory through the one bound store.

`build_command` is the in-container CLI contract (bootstrap + `python -m
simple_agent_lab.evals.in_container`); only container backends use it. The
run-directory convention is one ``<run_root>/<run_id>/<instance_id>/`` tree
with ``input/instance.json`` and ``out/{trajectory,prediction}.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_script
from .protocols import (
    INSTANCE_KEY,
    TRACE_KEY,
    ArtifactStore,
    ContainerBinding,
    ContainerBackend,
    RunArtifacts,
    RunSpec,
    Suite,
)

GENERIC_RUNNER_MODULE = "simple_agent_lab.evals.in_container"


@dataclass(frozen=True)
class RunPaths:
    """The standard per-instance directory tree."""

    root: Path
    input_dir: Path
    output_dir: Path
    trajectory_jsonl: Path


@dataclass(frozen=True)
class PreparedRun:
    """Inputs shared by blocking and detached backend entry points."""

    paths: RunPaths
    store: ArtifactStore
    binding: ContainerBinding
    spec: RunSpec


def safe_path_part(value: str) -> str:
    """Filesystem/Docker-safe form of an id, collision-free across distinct ids.

    Plain ids (alnum / ``_.-``) are returned unchanged. When sanitization would
    replace a character, a short hash of the *raw* value is appended so two ids
    that differ only in replaced characters (``a:b`` vs ``a_b``) map to distinct
    run dirs and container names instead of silently sharing one.
    """

    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    if not safe or safe in {".", ".."}:
        safe = "run"
    if safe != value:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return safe


def _safe_part(value: str) -> str:
    """Backward-compatible private name for :func:`safe_path_part`."""

    return safe_path_part(value)


def canonical_run_id(value: str) -> str:
    """Return the single path-safe representation used for a run namespace."""

    return safe_path_part(value)


def prepare_run_directory(*, run_root: Path, run_id: str, instance_id: str) -> RunPaths:
    """Create the input/out directories for one instance."""

    root = run_root.resolve() / canonical_run_id(run_id) / _safe_part(instance_id)
    input_dir = root / "input"
    output_dir = root / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        trajectory_jsonl=output_dir / TRACE_KEY.split("/")[-1],
    )


def prepare_new_run_directory(*, run_root: Path, run_id: str) -> Path:
    """Create a fresh run namespace, refusing to reuse existing artifacts."""

    root = run_root.resolve() / canonical_run_id(run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts: {root}. "
            "Choose a new --run-id; exact run resume is not supported."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def clear_run_outputs(paths: RunPaths) -> None:
    """Remove products from an earlier execution of the same run/instance."""

    if paths.output_dir.exists():
        shutil.rmtree(paths.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)


# Docker container names cap at 255 chars; long SWE-bench instance_ids + a long
# run_id could overflow. Keep a safety margin and, when over, truncate the joined
# name and append a short hash so distinct overflowing names stay distinct.
_MAX_CONTAINER_NAME = 200


def _run_root_namespace(run_root: Path) -> str:
    return hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:8]


def container_name(
    suite_name: str, instance_id: str, run_id: str, *, namespace: str = ""
) -> str:
    parts = [_safe_part(suite_name)]
    if namespace:
        parts.append(_safe_part(namespace))
    parts.extend((_safe_part(instance_id), _safe_part(run_id)))
    name = ".".join(parts)
    if len(name) > _MAX_CONTAINER_NAME:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: _MAX_CONTAINER_NAME - 9]}-{digest}"
    return name


def build_command(spec: RunSpec) -> tuple[str, ...]:
    """The container's main process: bootstrap + the generic in-container runner.

    Only container backends call this; `LocalProcessBackend` runs in-process and
    never builds a command.
    """

    runner_argv: list[str] = [
        "-m",
        spec.runner_module,
        "--container-module",
        spec.container_module,
        "--suite-name",
        spec.suite_name,
        "--instance-id",
        spec.instance_id,
        "--workdir",
        spec.launch_spec.workdir,
        "--max-turns",
        str(spec.max_turns),
        "--provider",
        spec.provider,
        "--api-kind",
        spec.api_kind,
    ]
    if spec.wall_time_seconds is not None:
        runner_argv += ["--wall-time-seconds", str(spec.wall_time_seconds)]
    script = bootstrap_script(
        runner_argv=tuple(runner_argv),
        install=spec.install,
        wheelhouse_mount=spec.wheelhouse_mount,
        package_extras=spec.package_extras,
    )
    return tuple(spec.launch_spec.shell) + (script,)


def _prepare_run(
    *,
    suite: Suite,
    instance: Mapping[str, Any],
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    provider: str,
    api_kind: str,
    max_turns: int,
    wall_time_seconds: float | None,
    provider_env: Mapping[str, str] | None,
    runner_module: str,
    install: bool,
    package_extras: tuple[str, ...],
    wheelhouse_mount: str | None,
    name: str | None,
) -> PreparedRun:
    instance_id = str(instance["instance_id"])
    paths = prepare_run_directory(
        run_root=run_root, run_id=run_id, instance_id=instance_id
    )
    clear_run_outputs(paths)
    launch_spec = suite.launch_spec(instance)
    bound = store.bind(paths.root)
    # Only the trusted oracle run receives the unredacted reference solution.
    record = dict(instance) if provider == "oracle" else suite.task_input(instance)
    bound.put(
        INSTANCE_KEY,
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )
    _stage_eval_inputs(suite, instance, bound)
    return PreparedRun(
        paths=paths,
        store=bound,
        binding=bound.container_binding(),
        spec=RunSpec(
            suite_name=suite.name,
            container_module=suite.container_module,
            instance_id=instance_id,
            launch_spec=launch_spec,
            max_turns=max_turns,
            provider=provider,
            api_kind=api_kind,
            provider_env=dict(provider_env or {}),
            runner_module=runner_module,
            install=install,
            package_extras=package_extras,
            wheelhouse_mount=wheelhouse_mount,
            run_name=name
            or container_name(
                suite.name,
                instance_id,
                run_id,
                namespace=_run_root_namespace(run_root),
            ),
            wall_time_seconds=wall_time_seconds,
        ),
    )


def run_suite_instance(
    *,
    suite: Suite,
    instance: Mapping[str, Any],
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    provider: str = "openai",
    api_kind: str = "openai-chat",
    max_turns: int = 75,
    wall_time_seconds: float | None = None,
    provider_env: Mapping[str, str] | None = None,
    runner_module: str = GENERIC_RUNNER_MODULE,
    install: bool = True,
    package_extras: tuple[str, ...] = (),
    wheelhouse_mount: str | None = None,
    name: str | None = None,
) -> RunArtifacts:
    """Run one instance and return where its artifacts landed.

    The agent-visible task input is written through `store` under ``input/``;
    the backend runs the suite's container half (reading that instance, writing
    ``out/result.json`` + ``out/trajectory.jsonl`` through the same store).
    ``out/result.json`` is the raw `extract_result` product; when the suite
    scores in the run environment (the container-half ``evaluate`` hook, enabled
    by staging ``eval_inputs``) its verdict is merged into that same file.
    Otherwise scoring is a follow-up run (an agent judge) or an external oracle
    (the official harness reading ``out/result.json``).
    """

    prepared = _prepare_run(
        suite=suite,
        instance=instance,
        store=store,
        run_root=run_root,
        run_id=run_id,
        max_turns=max_turns,
        provider=provider,
        api_kind=api_kind,
        provider_env=provider_env,
        runner_module=runner_module,
        install=install,
        package_extras=package_extras,
        wheelhouse_mount=wheelhouse_mount,
        name=name,
        wall_time_seconds=wall_time_seconds,
    )
    outcome = backend.run(prepared.spec, store=prepared.store, binding=prepared.binding)
    prepared.store.collect_outputs()

    return RunArtifacts(
        instance_id=prepared.spec.instance_id,
        run_dir=prepared.paths.root,
        trajectory_path=prepared.paths.trajectory_jsonl,
        status_code=outcome.status_code,
        logs=outcome.logs,
    )


def _stage_eval_inputs(
    suite: Suite, instance: Mapping[str, Any], bound: ArtifactStore
) -> None:
    """Stage gold scoring inputs (the "reuse" topology) under EVAL_KEY, if any.

    A suite's ``eval_inputs(instance)`` hands the run environment what its
    container-half ``evaluate`` needs (e.g. the official eval script) without
    putting it in the agent-visible instance. Written to a separate key so
    ``task_input`` still governs what the agent sees. ``eval_inputs`` returns
    ``None`` for the "separate" topology, in which case nothing is staged.
    """

    from .protocols import EVAL_KEY

    payload = suite.eval_inputs(instance)
    if not payload:
        return
    bound.put(
        EVAL_KEY,
        (json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8"),
    )
