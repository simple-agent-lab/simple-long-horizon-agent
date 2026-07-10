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
run-directory convention (ADR eval-output-directory-convention) is preserved: one
``<run_root>/<run_id>/<instance_id>/`` tree with ``input/instance.json`` and
``out/{trajectory,prediction}.jsonl``.
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
    ContainerBackend,
    RunArtifacts,
    RunSpec,
    Suite,
)

GENERIC_RUNNER_MODULE = "simple_agent_lab.evals.in_container"


@dataclass(frozen=True)
class RunPaths:
    """The standard per-instance directory tree (ADR eval-output-directory-convention)."""

    root: Path
    input_dir: Path
    output_dir: Path
    instance_json: Path
    trajectory_jsonl: Path
    prediction_jsonl: Path


def _safe_part(value: str) -> str:
    """Filesystem/Docker-safe form of an id, collision-free across distinct ids.

    Plain ids (alnum / ``_.-``) are returned unchanged. When sanitization would
    replace a character, a short hash of the *raw* value is appended so two ids
    that differ only in replaced characters (``a:b`` vs ``a_b``) map to distinct
    run dirs and container names instead of silently sharing one.
    """

    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    if safe != value:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return safe


def prepare_run_directory(*, run_root: Path, run_id: str, instance_id: str) -> RunPaths:
    """Create the input/out dirs for one instance (ADR eval-output-directory-convention layout)."""

    root = run_root.resolve() / _safe_part(run_id) / _safe_part(instance_id)
    input_dir = root / "input"
    output_dir = root / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        instance_json=input_dir / "instance.json",
        trajectory_jsonl=output_dir / TRACE_KEY.split("/")[-1],
        prediction_jsonl=output_dir / "prediction.jsonl",
    )


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

    instance_id = str(instance["instance_id"])
    paths = prepare_run_directory(
        run_root=run_root, run_id=run_id, instance_id=instance_id
    )
    clear_run_outputs(paths)
    launch_spec = suite.launch_spec(instance)

    # The agent must never see gold/private fields, so they are stripped here.
    # Oracle mode is the trusted exception: it *applies* the reference solution,
    # so it needs the unredacted record (e.g. the gold patch) in the store.
    record = dict(instance) if provider == "oracle" else suite.task_input(instance)
    bound = store.bind(paths.root)
    bound.put(
        INSTANCE_KEY,
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )
    _stage_eval_inputs(suite, instance, bound)
    binding = bound.container_binding()

    spec = RunSpec(
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
    )
    outcome = backend.run(spec, store=bound, binding=binding)
    bound.collect_outputs()

    return RunArtifacts(
        instance_id=instance_id,
        run_dir=paths.root,
        trajectory_path=paths.trajectory_jsonl,
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
