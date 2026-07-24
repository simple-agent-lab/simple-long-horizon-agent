"""OneMillion-Bench as a `Suite` (ADR 0017).

This is the *host half*: it maps a OneMillion-Bench case onto one `Suite` whose
launch shape rides along as ``launch_spec`` data, drops the rubrics before the
agent sees the case (``task_input``), and stages those rubrics as gold scoring
inputs (``eval_inputs``) so the container-half ``evaluate`` hook grades the
answer in the run environment.

The *container half* (``build_agent`` / ``build_task`` / ``extract_result`` /
``evaluate``) ships in the wheel at
``simple_agent_lab.evals.suites.onemillion.container`` and is driven by the
generic in-container runner — so no files are copied into the run environment.

Unlike SWE-bench, OneMillion-Bench needs no Docker image: it is a tool-free Q&A
generation graded by a judge model, so it runs through ``LocalProcessBackend``
by default. ``launch_spec.image`` is therefore only a fallback for the
containerized backends; the in-process backend ignores it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec

from . import harness


class OneMillionSuite:
    """`Suite` for OneMillion-Bench rubric-graded Q&A cases."""

    name = harness.SUITE_NAME
    container_module = "simple_agent_lab.evals.suites.onemillion.container"

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        workdir: str = harness.DEFAULT_WORKDIR,
        in_env_scoring: bool = True,
    ) -> None:
        self.image = image
        self.workdir = workdir
        # Scoring is the point of this suite and there is no separate official
        # harness, so in-environment rubric judging is on by default. Turn it off
        # to capture answers only (e.g. to grade later or with a different judge).
        self.in_env_scoring = in_env_scoring

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        del instance  # every case launches the same way (no per-case image)
        return LaunchSpec(image=self.image, workdir=self.workdir)

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return harness.sanitized_instance(dict(instance))

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        """Stage the case's rubrics for the in-environment judge (gold the agent
        never sees). Returns ``None`` when scoring is disabled or the case has no
        rubrics, which leaves the ``evaluate`` hook off."""

        if not self.in_env_scoring:
            return None
        return harness.eval_payload(dict(instance))


class OneMillionWorkflowSuite(OneMillionSuite):
    """`OneMillionSuite` whose generation runs a multi-agent *workflow*.

    Identical host behavior (task sanitization, rubric staging, judge scoring) —
    only the container half differs: it points at ``workflow_container``, whose
    facade ``build_agent`` runs the workflow named by the ``OMB_WORKFLOW`` env
    var (reflection / planner_executor / parallel / chain / routing / single).
    """

    container_module = "simple_agent_lab.evals.suites.onemillion.workflow_container"


class OneMillionDynamicWorkflowSuite(OneMillionSuite):
    """`OneMillionSuite` whose generation runs agent-written JavaScript workflow."""

    container_module = (
        "simple_agent_lab.evals.suites.onemillion.dynamic_workflow_container"
    )
