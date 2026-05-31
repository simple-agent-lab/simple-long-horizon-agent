"""Generic orchestration for one eval instance.

`run_suite_instance(...)` wires a `Suite` + `ContainerBackend` + `ArtifactStore`
together: it resolves the launch plan, seeds the instance into the store, hands a
`RunSpec` to the backend, and shapes the prediction from the result. It has no
`if pro:` branches and makes no Docker calls of its own — the suite supplies the
plan as data, and the backend owns the run.

The *same* call runs locally or across machines by swapping the backend:
`LocalProcessBackend` (in-process, no Docker — local dev) /
`LocalDockerBackend` (one machine) / a remote backend (multi-machine). The
suite's container half runs identically either way, because every backend reads
the instance and writes the result/trajectory through the one bound store.

`build_command` is the in-container CLI contract (bootstrap + `python -m
simple_agent_lab.evals.in_container`); only container backends use it. The
run-directory convention (ADR 0016) is preserved: one
``<run_root>/<run_id>/<instance_id>/`` tree with ``input/instance.json`` and
``out/{trajectory,prediction}.jsonl``.
"""

from __future__ import annotations

import hashlib
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
    RunArtifacts,
    RunSpec,
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


def build_command(spec: RunSpec) -> tuple[str, ...]:
    """The container's main process: bootstrap + the generic in-container runner.

    Only container backends call this; `LocalProcessBackend` runs in-process and
    never builds a command.
    """

    runner_argv = (
        "-m",
        GENERIC_RUNNER_MODULE,
        "--container-module",
        spec.container_module,
        "--suite-name",
        spec.suite_name,
        "--instance-id",
        spec.instance_id,
        "--workdir",
        spec.plan.workdir,
        "--max-turns",
        str(spec.max_turns),
        "--provider",
        spec.provider,
        "--api-kind",
        spec.api_kind,
    )
    script = bootstrap_script(
        runner_argv=runner_argv,
        install=spec.install,
        wheelhouse_mount=spec.wheelhouse_mount,
    )
    return tuple(spec.plan.shell) + (script,)


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
    name: str | None = None,
) -> RunArtifacts:
    """Run one instance and return where its artifacts landed.

    The sanitized instance is written through `store` under ``input/``; the
    backend runs the suite's container half (reading that instance, writing
    ``out/result.json`` + ``out/trajectory.jsonl`` through the same store). After
    the run the scorer-facing ``out/prediction.jsonl`` is shaped from the result
    via `suite.prediction_record`, so prediction formatting stays host-side with
    the rest of the suite config.
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

    spec = RunSpec(
        suite_name=suite.name,
        container_module=suite.container_module,
        instance_id=instance_id,
        plan=plan,
        max_turns=max_turns,
        provider=provider,
        api_kind=api_kind,
        provider_env=dict(provider_env or {}),
        install=install,
        wheelhouse_mount=wheelhouse_mount,
        run_name=name or container_name(suite.name, instance_id, run_id),
    )
    outcome = backend.run(spec, store=bound, binding=binding)
    bound.collect_outputs()

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
        status_code=outcome.status_code,
        logs=outcome.logs,
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
