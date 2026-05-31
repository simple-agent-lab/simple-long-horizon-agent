"""Generic orchestration for one containerized eval instance.

`run_suite_instance(...)` wires a `Suite` + `ContainerBackend` +
`ArtifactTransport` together and drives the lifecycle. It has no `if pro:`
branches and makes no Docker calls of its own — those live in the suite (as
data via `ContainerPlan`) and the backend respectively. Swapping local Docker
for a cloud backend, or a bind mount for copy-out, changes the arguments here,
not the body.

The run-directory convention (ADR 0016) is preserved: one
``<run_root>/<run_id>/<instance_id>/`` tree with ``input/instance.json`` and
``out/{trajectory,prediction}.jsonl``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocols import (
    ArtifactTransport,
    ContainerBackend,
    RunArtifacts,
    StagedFile,
    Suite,
)


@dataclass(frozen=True)
class RunPaths:
    """The standard per-instance directory tree (ADR 0016)."""

    root: Path
    input_dir: Path
    output_dir: Path
    instance_json: Path
    trajectory_jsonl: Path
    prediction_jsonl: Path


def _safe_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)


def prepare_run_directory(
    *,
    suite: Suite,
    run_root: Path,
    run_id: str,
    instance: Mapping[str, Any],
) -> RunPaths:
    """Create the input/out dirs and write the suite-sanitized instance record."""

    instance_id = str(instance["instance_id"])
    root = run_root.resolve() / _safe_part(run_id) / _safe_part(instance_id)
    input_dir = root / "input"
    output_dir = root / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_json = input_dir / "instance.json"
    instance_json.write_text(
        json.dumps(
            suite.sanitize_instance(instance), ensure_ascii=False, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return RunPaths(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        instance_json=instance_json,
        trajectory_jsonl=output_dir / "trajectory.jsonl",
        prediction_jsonl=output_dir / "prediction.jsonl",
    )


def container_name(suite_name: str, instance_id: str, run_id: str) -> str:
    return f"{_safe_part(suite_name)}.{_safe_part(instance_id)}.{_safe_part(run_id)}"


def run_suite_instance(
    *,
    suite: Suite,
    instance: Mapping[str, Any],
    backend: ContainerBackend,
    transport: ArtifactTransport,
    command: tuple[str, ...],
    run_root: Path,
    run_id: str,
    env: Mapping[str, str] | None = None,
    extra_inputs: tuple[StagedFile, ...] = (),
    name: str | None = None,
    keep_container: bool = False,
) -> RunArtifacts:
    """Run one instance and return where its artifacts landed.

    `command` is the container's main process (typically the bootstrap script
    from `bootstrap.py` plus the in-container runner invocation). `extra_inputs`
    are out-of-tree files the transport must place in the container (the runner
    module, support files, an optional uv binary); the run directory itself
    reaches the container via the transport's mounts.
    """

    instance_id = str(instance["instance_id"])
    plan = suite.container_plan(instance)
    paths = prepare_run_directory(
        suite=suite, run_root=run_root, run_id=run_id, instance=instance
    )

    handle = backend.create(
        name=name or container_name(suite.name, instance_id, run_id),
        plan=plan,
        command=command,
        env=dict(env or {}),
        mounts=transport.mounts(paths.root),
    )
    try:
        transport.stage_inputs(handle, run_dir=paths.root, files=extra_inputs)
        handle.start()
        status_code = handle.wait()
        logs = handle.logs()
        transport.collect_outputs(handle, run_dir=paths.root)
    finally:
        if not keep_container:
            handle.remove()

    return RunArtifacts(
        instance_id=instance_id,
        run_dir=paths.root,
        trajectory_path=paths.trajectory_jsonl,
        prediction_path=paths.prediction_jsonl,
        status_code=status_code,
        logs=logs,
    )
