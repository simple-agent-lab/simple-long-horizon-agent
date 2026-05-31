"""Generic orchestration for one containerized eval instance.

`run_suite_instance(...)` wires a `Suite` + `ContainerBackend` + `ArtifactStore`
together and drives the lifecycle. It has no `if pro:` branches and makes no
Docker calls of its own — those live in the suite (as data via `ContainerPlan`)
and the backend respectively. Swapping local Docker for a cloud backend, or a
bind mount for an HTTP/object store, changes the arguments here, not the body.

The framework builds the container command itself (bootstrap + the generic
in-container runner invoked as ``python -m simple_agent_lab.evals.in_container``),
so callers never hand-assemble argv or copy a runner in. The in-container CLI
contract stays internal.

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

from .bootstrap import bootstrap_script
from .protocols import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    ArtifactStore,
    ContainerBackend,
    ContainerPlan,
    RunArtifacts,
    Suite,
)

GENERIC_RUNNER_MODULE = "simple_agent_lab.evals.in_container"


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


def prepare_run_directory(*, run_root: Path, run_id: str, instance_id: str) -> RunPaths:
    """Create the input/out dirs for one instance (ADR 0016 layout)."""

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


def container_name(suite_name: str, instance_id: str, run_id: str) -> str:
    return f"{_safe_part(suite_name)}.{_safe_part(instance_id)}.{_safe_part(run_id)}"


def build_command(
    *,
    suite: Suite,
    plan: ContainerPlan,
    instance_id: str,
    max_turns: int,
    provider: str,
    api_kind: str,
    install: bool,
    wheelhouse_mount: str | None,
) -> tuple[str, ...]:
    """The container's main process: bootstrap + the generic in-container runner."""

    runner_argv = (
        "-m",
        GENERIC_RUNNER_MODULE,
        "--container-module",
        suite.container_module,
        "--suite-name",
        suite.name,
        "--instance-id",
        instance_id,
        "--workdir",
        plan.workdir,
        "--max-turns",
        str(max_turns),
        "--provider",
        provider,
        "--api-kind",
        api_kind,
    )
    script = bootstrap_script(
        runner_argv=runner_argv, install=install, wheelhouse_mount=wheelhouse_mount
    )
    return tuple(plan.shell) + (script,)


def run_suite_instance(
    *,
    suite: Suite,
    instance: Mapping[str, Any],
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    model_name: str = "simple-agent-lab",
    provider: str = "openai",
    api_kind: str = "openai-chat",
    max_turns: int = 75,
    provider_env: Mapping[str, str] | None = None,
    install: bool = True,
    wheelhouse_mount: str | None = None,
    command_override: tuple[str, ...] | None = None,
    name: str | None = None,
    keep_container: bool = False,
) -> RunArtifacts:
    """Run one instance and return where its artifacts landed.

    The sanitized instance is written through `store` under ``input/``; the
    container reads it, runs the agent, and writes ``out/result.json`` +
    ``out/trajectory.jsonl`` back through the same store. After the run the
    scorer-facing ``out/prediction.jsonl`` is shaped from the result via
    `suite.prediction_record`, so prediction formatting stays host-side with the
    rest of the suite config.
    """

    instance_id = str(instance["instance_id"])
    plan = suite.container_plan(instance)
    paths = prepare_run_directory(
        run_root=run_root, run_id=run_id, instance_id=instance_id
    )

    bound = store.bind(paths.root)
    bound.put(
        INSTANCE_KEY,
        (
            json.dumps(
                suite.sanitize_instance(instance), ensure_ascii=False, sort_keys=True
            )
            + "\n"
        ).encode("utf-8"),
    )
    binding = bound.container_binding()

    command = command_override or build_command(
        suite=suite,
        plan=plan,
        instance_id=instance_id,
        max_turns=max_turns,
        provider=provider,
        api_kind=api_kind,
        install=install,
        wheelhouse_mount=wheelhouse_mount,
    )
    env = {**dict(provider_env or {}), **binding.env}

    handle = backend.create(
        name=name or container_name(suite.name, instance_id, run_id),
        plan=plan,
        command=command,
        env=env,
        mounts=binding.mounts,
        add_hosts=binding.add_hosts,
    )
    try:
        handle.start()
        status_code = handle.wait()
        logs = handle.logs()
        bound.collect_outputs()
    finally:
        if not keep_container:
            handle.remove()

    _shape_prediction(
        suite=suite,
        instance=instance,
        model_name=model_name,
        store=bound,
        prediction_path=paths.prediction_jsonl,
    )

    return RunArtifacts(
        instance_id=instance_id,
        run_dir=paths.root,
        trajectory_path=paths.trajectory_jsonl,
        prediction_path=paths.prediction_jsonl,
        status_code=status_code,
        logs=logs,
    )


def _shape_prediction(
    *,
    suite: Suite,
    instance: Mapping[str, Any],
    model_name: str,
    store: ArtifactStore,
    prediction_path: Path,
) -> None:
    """Write ``prediction.jsonl`` from the container's result, if present."""

    try:
        raw = store.get(RESULT_KEY)
    except (FileNotFoundError, OSError):
        return
    result = json.loads(raw.decode("utf-8") or "{}")
    prediction = suite.prediction_record(instance, model_name=model_name, result=result)
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
    )
