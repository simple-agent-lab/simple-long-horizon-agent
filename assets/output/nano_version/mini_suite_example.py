"""A tiny suite showing the host-half / container-half split.

This file is not trying to be a real benchmark. It demonstrates what a new
suite contributes: launch data, visible task input, and hidden eval input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nano_eval import LaunchSpec, LocalDirStore, LocalProcessBackend, run_suite_instance


class FixPrintSuite:
    name = "fix-print"
    container_module = "mini_suite_example"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(
            image="python:3.12-slim",
            workdir="/workspace",
        )

    def task_input(self, instance: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "instance_id": instance["instance_id"],
            "prompt": "Edit app.py so it prints the expected answer.",
            "files": {"app.py": "print(41)"},
        }

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"expected_stdout": instance["expected_stdout"]}


# Container half in the real framework:
#   build_task(instance, *, workdir) -> model-visible task
#   extract_result(workspace, instance) -> raw result
#   evaluate(workspace, instance, *, context) -> optional in-env score


def demo_shape() -> Path:
    """This is the shape a caller uses; backend choice is the switch."""

    return run_suite_instance(
        suite=FixPrintSuite(),
        instance={"instance_id": "case-001", "expected_stdout": "42"},
        backend=LocalProcessBackend(),
        store=LocalDirStore(Path("runs")),
        run_root=Path("runs"),
        run_id="nano-demo",
    )

