"""Candidate agent — the container half of a trivial "solver" suite.

Mapped onto the current framework exactly like SWE-bench: `build_task` makes the
task, the generic runner drives the agent loop (recording a trajectory), and
`extract_result` reads the agent's effect on the *workspace* (not its chat) to
form the product. Here the "product" is a listing of the workspace; in a real
suite it would be a `git diff`, an answer file, etc.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec, ContainerPlan


class CandidateSuite:
    """Host half — trivial, since in-process runs ignore image/workdir here."""

    name = "candidate"
    container_module = "examples.agent_judge.candidate"

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan:
        return ContainerPlan(image="(in-process)", workdir="(in-process)")

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def prediction_record(
        self, instance: Mapping[str, Any], *, model_name: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {"instance_id": str(instance["instance_id"]), **result}


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="candidate",
        role="Solve the task with bash.",
        system_prompt=(
            "You are a candidate agent. Investigate the workspace with bash, "
            "then give a short final answer."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    return (
        f"Problem: {instance['problem']}\n"
        f"Workspace: {workdir}\n"
        "Use bash to look around, then summarize what you found."
    )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace)
    files = sorted(p.name for p in workspace.iterdir()) if workspace.exists() else []
    return {"answer": f"workspace files: {files}"}
