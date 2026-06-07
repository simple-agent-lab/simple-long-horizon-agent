"""ClawBench-Tribe container half: build task + extract result.

Simple LLM reasoning tests. The agent answers a question; scoring is done
host-side by reading the trajectory.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="tribe_agent",
        role="Answer questions accurately and follow instructions precisely.",
        system_prompt=(
            "You are a helpful assistant. Answer questions directly and accurately. "
            "Follow instructions precisely. Do not add unnecessary commentary."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    return str(instance["prompt"])


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {"instance_id": instance["instance_id"], "status": "completed"}
