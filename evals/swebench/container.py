"""SWE-bench container half (ADR 0017): the "two functions" a suite supplies.

The generic in-container runner (`simple_agent_lab.evals.in_container`) owns
the agent loop, retry, and trace push. This module supplies only what is
SWE-bench-specific and runs *inside* the image:

- `build_task(instance, *, workdir)` — the model-visible task.
- `extract_result(workspace, instance, *, context)` — the run's product
  (`{"model_patch": diff}`).
- `prepare(workspace, instance)` — optional pre-run setup: snapshot a baseline
  commit and install generated-file ignore rules so the diff stays clean. Its
  return value is threaded back into `extract_result` as `context`.
- `agent_spec()` — optional: the SWE-bench prompt/role and the bash vs
  bash_task flavor (from the ``AGENT_FLAVOR`` env var).

It re-expresses the SWE-bench-specific pieces that currently live in
`in_container_runner` against the generic runner's surface; the prompt text and
task builder stay sourced from `in_container_runner` so there is one source of
truth until that legacy launcher is retired.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.agents.bash_task import BASH_TASK_EXPLORER_ADDENDUM
from simple_agent_lab.evals.protocols import AgentSpec

from . import in_container_runner as icr
from .patch_extract import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
)

AGENT_FLAVOR_ENV = "AGENT_FLAVOR"


def agent_spec() -> AgentSpec:
    """SWE-bench agent config; flavor from ``AGENT_FLAVOR`` (bash | bash_task)."""

    flavor = os.environ.get(AGENT_FLAVOR_ENV, "bash").strip() or "bash"
    system_prompt = icr.AGENT_SYSTEM_PROMPT
    if flavor == "bash_task":
        system_prompt = system_prompt + "\n\n" + BASH_TASK_EXPLORER_ADDENDUM
    return AgentSpec(
        name=icr.AGENT_NAME,
        role=icr.AGENT_ROLE,
        system_prompt=system_prompt,
        flavor=flavor,
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    return icr.task_from_instance(dict(instance), workdir=workdir)


def prepare(workspace: Path, instance: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot a baseline commit + ignore rules before the agent edits."""

    workspace = Path(workspace)
    record = dict(instance)
    language = instance_language(record)
    baseline = prepare_baseline_commit(workspace, language=language) or (
        instance_base_commit(record)
    )
    return {"language": language, "baseline_commit": baseline}


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the staged `git diff` as the SWE-bench prediction patch."""

    context = context or {}
    record = dict(instance)
    language = str(context.get("language") or instance_language(record))
    commit = context.get("baseline_commit") or instance_base_commit(record)
    return {"model_patch": git_diff(Path(workspace), language=language, commit=commit)}
