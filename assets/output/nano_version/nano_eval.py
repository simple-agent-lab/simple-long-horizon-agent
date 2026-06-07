"""Nano eval framework.

The real project uses this shape for Docker / SWE-bench style evaluation:

    Suite + ContainerBackend + ArtifactStore -> run_suite_instance(...)

The point is not Docker itself. The point is that Docker is only one backend
behind the same protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


INSTANCE_KEY = "input/instance.json"
EVAL_KEY = "input/eval.json"
RESULT_KEY = "out/result.json"
TRACE_KEY = "out/trajectory.jsonl"


@dataclass(frozen=True)
class LaunchSpec:
    """Backend-agnostic launch data.

    Docker needs image/workdir/shell. Local process may ignore image. A future
    k8s backend could translate the same data into a Job manifest.
    """

    image: str
    workdir: str
    shell: tuple[str, ...] = ("bash", "-lc")
    entrypoint: str | None = None


@dataclass(frozen=True)
class RunSpec:
    suite_name: str
    container_module: str
    instance_id: str
    launch: LaunchSpec
    max_turns: int = 50
    provider: str = "openai"


@dataclass(frozen=True)
class RunOutcome:
    status_code: int
    logs: str = ""


class Suite(Protocol):
    """Benchmark-specific host half."""

    name: str
    container_module: str

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec: ...

    def task_input(self, instance: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None: ...


class ArtifactStore(Protocol):
    """One byte store for inputs, outputs, and live trace."""

    def bind(self, run_dir: Path) -> "ArtifactStore": ...

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def collect_outputs(self) -> None: ...


class ContainerBackend(Protocol):
    """Where the run executes."""

    def run(self, spec: RunSpec, *, store: ArtifactStore) -> RunOutcome: ...


class LocalDirStore:
    """Small store: keys become files under one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def bind(self, run_dir: Path) -> "LocalDirStore":
        return LocalDirStore(run_dir)

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def collect_outputs(self) -> None:
        # A bind-mounted local directory already has the outputs.
        return None


class LocalProcessBackend:
    """Runs the container half in this Python process.

    This is the teaching/dev backend: same Suite shape, no Docker. Heavy suites
    may still need Docker because the benchmark environment is the product.
    """

    def run(self, spec: RunSpec, *, store: ArtifactStore) -> RunOutcome:
        instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))

        # Real code imports spec.container_module and calls:
        #   build_task(instance, workdir=...)
        #   run(agent, task)
        #   extract_result(workspace, instance)
        fake_result = {
            "instance_id": instance["instance_id"],
            "backend": "local-process",
            "note": "Here the real in-container runner would run the agent.",
        }
        store.put(RESULT_KEY, json.dumps(fake_result, indent=2).encode("utf-8"))
        store.put(TRACE_KEY, b'{"event":"nano_trace_placeholder"}\n')
        return RunOutcome(status_code=0)


class LocalDockerBackend:
    """Sketch only: Docker is an implementation detail of ContainerBackend."""

    def run(self, spec: RunSpec, *, store: ArtifactStore) -> RunOutcome:
        command = build_container_command(spec)
        return RunOutcome(
            status_code=0,
            logs=(
                "Would create Docker container with "
                f"image={spec.launch.image!r}, command={command!r}. "
                "The suite and runner do not change."
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
    max_turns: int = 50,
) -> Path:
    """One generic orchestration function.

    Notice what is absent: no SWE-bench branch, no Docker branch, no scoring
    special case. Suite shapes data; backend runs; store carries bytes.
    """

    instance_id = str(instance["instance_id"])
    run_dir = run_root / run_id / instance_id
    bound = store.bind(run_dir)

    # Hide gold/private fields from the agent-visible input.
    visible = dict(suite.task_input(instance))
    bound.put(INSTANCE_KEY, json.dumps(visible, indent=2).encode("utf-8"))

    gold = suite.eval_inputs(instance)
    if gold:
        bound.put(EVAL_KEY, json.dumps(dict(gold), indent=2).encode("utf-8"))

    spec = RunSpec(
        suite_name=suite.name,
        container_module=suite.container_module,
        instance_id=instance_id,
        launch=suite.launch_spec(instance),
        max_turns=max_turns,
    )
    backend.run(spec, store=bound)
    bound.collect_outputs()
    return run_dir


def build_container_command(spec: RunSpec) -> tuple[str, ...]:
    """The Docker backend's command shape.

    In the real project this bootstraps a wheel and invokes:
        python -m simple_agent_lab.evals.in_container
    """

    return (
        *spec.launch.shell,
        "python -m nano_eval_in_container "
        f"--container-module {spec.container_module} "
        f"--instance-id {spec.instance_id} "
        f"--workdir {spec.launch.workdir}",
    )

