"""SWE-bench container half (ADR 0017): the two functions a suite supplies.

The generic in-container runner (`simple_agent_lab.evals.in_container`) owns the
agent loop, retry, and trace push. This module supplies only what is
SWE-bench-specific and runs *inside* the image:

- `build_task(instance, *, workdir)` — the model-visible task.
- `extract_result(workspace, instance, *, context)` — the run's product
  (`{"model_patch": diff}`).
- `prepare(workspace, instance)` — optional pre-run setup: snapshot a baseline
  commit and install generated-file ignore rules so the diff stays clean. Its
  return value is threaded back into `extract_result` as `context`.
- `agent_spec()` — optional: the SWE-bench prompt/role and the bash vs
  bash_task flavor (from the ``AGENT_FLAVOR`` env var).

It imports only the standard library and the installed wheel (`agents`,
`evals.protocols`, and the sibling `patch` module), so it works inside any
SWE-bench image with no copied files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.agents.bash_task import BASH_TASK_EXPLORER_ADDENDUM
from simple_agent_lab.evals.protocols import AgentSpec

from .patch import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
)

AGENT_FLAVOR_ENV = "AGENT_FLAVOR"

AGENT_NAME = "swebench_agent"
AGENT_ROLE = (
    "Work in the local SWE-bench repository. Use bash for inspection, edits, "
    "and focused tests, then return a concise final note."
)
AGENT_SYSTEM_PROMPT = (
    "You are a software engineer interacting with a SWE-bench instance "
    "container through the bash tool. Each bash call runs in a fresh shell "
    "rooted at the workspace, so include any cd or env setup in the command "
    "and use non-interactive flags (`-y`, `--no-pager`, avoid `vi`/`nano`). "
    "Independent read-only bash calls may run in parallel; never run parallel "
    "writes against the same file. Work from evidence: inspect, reproduce, "
    "edit, verify — make a fix that is general and consistent with the "
    "codebase. Keep command output focused. When the repository is patched, "
    "return a short final summary; the harness collects git diff separately."
)


def agent_spec() -> AgentSpec:
    """SWE-bench agent config; flavor from ``AGENT_FLAVOR`` (bash | bash_task)."""

    flavor = os.environ.get(AGENT_FLAVOR_ENV, "bash").strip() or "bash"
    system_prompt = AGENT_SYSTEM_PROMPT
    if flavor == "bash_task":
        system_prompt = system_prompt + "\n\n" + BASH_TASK_EXPLORER_ADDENDUM
    return AgentSpec(
        name=AGENT_NAME, role=AGENT_ROLE, system_prompt=system_prompt, flavor=flavor
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the model-visible task for the in-container agent."""

    problem = _optional(
        instance.get("problem_statement")
        or instance.get("problem")
        or instance.get("description")
        or ""
    )
    requirements = _optional(instance.get("requirements"))
    interface = _optional(instance.get("interface"))
    lines = [
        "Solve this SWE-bench instance.",
        "",
        "## Environment",
        "- You are running inside the SWE-bench container.",
        f"- The bash tool runs locally in {workdir}.",
        "- A full Linux shell is available; install missing tools only if strictly needed.",
        "- Always pass non-interactive flags (`-y`, `--no-pager`); avoid editors that wait for input.",
        "",
        "## What to modify",
        "- MODIFY: regular source files in the repository.",
        "- DO NOT MODIFY: tests, reproduction scripts you create, configuration files",
        "  (pyproject.toml, setup.cfg, tox.ini, etc.) unless code evidence shows the fix",
        "  belongs there.",
        "- Keep temporary reproduction helpers out of the final diff (write them under",
        "  `/tmp/` or delete them before you stop).",
        "",
        "## Workflow",
        "1. Locate the relevant code. Prefer parallel read-only commands",
        "   (`grep -rn`, `find`, `sed -n 'A,Bp'`) over reading whole files.",
        "2. Reproduce the reported behavior with a tiny script when practical.",
        "3. Edit the smallest set of source files needed for a general fix.",
        "4. Re-run the reproduction. Then run a focused subset of existing tests",
        "   (single file or `-k pattern`) and explain if any are unavailable.",
        "5. Stop as soon as the fix is in place and verified. Do not keep exploring",
        "   once you can describe the change.",
        "",
        "## Final answer",
        "Return a short summary of the files you changed and how you verified the fix.",
        "Do NOT paste the patch — the harness collects `git diff` separately.",
        "",
        "## Problem statement",
        problem,
    ]
    if requirements:
        lines.extend(["", "requirements:", requirements])
    if interface:
        lines.extend(["", "interface:", interface])
    return "\n".join(lines)


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


def _optional(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return ""
    if text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, str):
            return decoded.strip()
    return text
