"""A small but *real* "fix the bug" suite — the reference shape for a benchmark.

Mapped onto the current framework exactly like SWE-bench: `build_task` makes the
task, the generic runner drives the agent loop (recording a trajectory), and
`extract_result` reads the agent's effect on the *workspace* (not its chat) to
form the product. Here the product is the program's actual output: the agent edits
`app.py` with bash, and `extract_result` *runs* it and captures stdout — the same
"edit code, then execute to check" shape SWE-bench uses, just one file instead of
a repo. It needs a real model (the agent must locate and fix the bug).

This suite scores *in the run environment* (ADR collapse-scorer-seam-into-run-primitive): the gold answer rides on
the instance as ``expected``; `task_input` *hides* it from the agent; `eval_inputs`
stages it under EVAL_KEY (gold the agent never sees); and the container-half
`evaluate` hook runs the program and compares its output against the gold, merging
a verdict into ``result.json``. The verdict lives next to the product, so there is
no separate scoring phase to run.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec, LaunchSpec

APP_FILE = "app.py"


class ExampleBenchSuite:
    """Host half. `launch_spec` declares the container shape (image + workdir) a
    container backend would use; `LocalProcessBackend` supplies its own workspace
    and ignores them, so the *same* suite runs both ways unchanged."""

    name = "candidate"
    container_module = "examples.bench_suite.candidate"
    image = "python:3.12-slim"
    workdir = "/workspace"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image=self.image, workdir=self.workdir)

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        # Hide the gold ``expected`` output so the agent must fix the code, not copy.
        return {k: v for k, v in instance.items() if k != "expected"}

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        # Stage the gold output so the `evaluate` hook can grade in place. The
        # agent never sees this (it goes under EVAL_KEY, not the task input);
        # staging it is also the toggle that turns the hook on.
        return {"expected": instance.get("expected", "")}


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="candidate",
        role="Fix the bug with bash.",
        system_prompt=(
            "You are a software-fixing agent. Investigate the workspace with bash "
            f"(cat, grep, sed, etc.), edit {APP_FILE} to fix the bug, and verify "
            f"by running it. Make the smallest change that works."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    return (
        f"Problem: {instance['problem']}\n"
        f"Workspace: {workdir} (the file to fix is {APP_FILE})\n"
        f"Edit {APP_FILE} in place with bash, then run it to confirm the output."
    )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {"answer": _program_output(workspace)}


def evaluate(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score in the run environment: does the fixed program print the gold output?

    The gold rides in ``context["eval"]`` (staged by `eval_inputs`, never shown to
    the agent). We run the program the agent edited — exactly what `extract_result`
    did — and grade its output; the runner merges this verdict into ``result.json``
    next to the product, so there is no separate scoring phase.
    """

    expected = str((context or {}).get("eval", {}).get("expected", ""))
    answer = _program_output(workspace)
    passed = bool(expected) and answer == expected
    return {"passed": passed, "score": 1.0 if passed else 0.0, "expected": expected}


def _program_output(workspace: Path) -> str:
    """Run the workspace's `app.py` and return its stdout (the run's product)."""

    app = Path(workspace) / APP_FILE
    if not app.exists():
        return ""
    proc = subprocess.run(
        [sys.executable, str(app)],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()
