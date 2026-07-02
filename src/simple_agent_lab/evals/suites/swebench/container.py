"""SWE-bench container half (ADR generic-containerized-eval-framework): the two functions a suite supplies.

The generic in-container runner (`simple_agent_lab.evals.in_container`) owns the
agent loop, retry, and trace push. This module supplies only what is
SWE-bench-specific and runs *inside* the image:

- `build_task(instance, *, workdir)` — the model-visible task.
- `extract_result(workspace, instance, *, context)` — the run's product
  (`{"model_patch": diff}`).
- `prepare(workspace, instance)` — optional pre-run setup: snapshot a baseline
  commit and install generated-file ignore rules so the diff stays clean. Its
  return value is threaded back into `extract_result` as `context`.
- `apply_oracle(workspace, instance)` — optional: apply the gold patch instead
  of running a model, for the framework's deterministic oracle self-check.
- `evaluate(workspace, instance, *, context)` — optional: in-environment scoring
  (ADR collapse-scorer-seam-into-run-primitive). Runs the host-staged official
  eval script in the run environment and captures its log into ``result.json``;
  the host turns that into a verdict via `evaluate_predictions.reuse_eval_row`
  (the official grader needs the gold test spec, which lives host-side).
- `agent_spec()` / `build_agent()` — the single agent seam, selected by one
  ``AGENT_FLAVOR`` env var. Simple flavors (``bash`` | ``bash_task`` |
  ``bash_task_read`` | ``bash_skills``) are built by the generic runner's
  ``agent_spec`` path (so they keep memory hooks); the multi-agent
  *workflow arms* (``loop`` | ``goal`` | ``pdr``) are built by the shared
  `agents.flavors` workflow builder. This suite's ``build_agent`` only passes
  SWE-bench prompt text, workspace cleanup, and trace-recording callbacks.

Why workflow arms live behind a thin facade (and worktrees)
-----------------------------------------------------------
An arm (``loop`` / ``goal`` / ``pdr``) runs a whole multi-agent choreography to produce one
patch, so `agents.flavors.build_flavor_agent` returns a facade
``Agent`` whose single ``generate`` runs the arm, leaves edits in the workspace,
and returns a short final note — the generic outer loop runs it once. ``pdr``
fans out concurrent attempts that each edit disk; if they shared one checkout
they'd clobber each other and the ``git diff`` would be garbage, so every attempt
gets its own ``git worktree`` (off the baseline commit, outside the workspace)
and resets it to baseline in an ``init_state`` hook — keeping rounds independent,
seeded only by the brief.

It imports only the standard library and the installed wheel (`agents`,
`evals.*`, `workflow`, `trace`, and the sibling `patch` module), so it works
inside any SWE-bench image with no copied files.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import simple_agent_lab.config as config
from simple_agent_lab.agent_flavors import (
    AGENT_FLAVORS,
    SIMPLE_AGENT_FLAVORS,
    WORKFLOW_AGENT_FLAVORS,
    flavor_from_env,
)
from simple_agent_lab.agents.flavors import (
    ArtifactPut,
    build_flavor_agent,
)
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.core import Agent
from simple_agent_lab.evals.chain import start_chain_state
from simple_agent_lab.evals.protocols import AgentSpec
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent

from .patch import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
    update_info_exclude,
)

# Back-compatible flavor-name aliases for older scripts/tests. The source of
# truth lives in `simple_agent_lab.agent_flavors`.
SIMPLE_FLAVORS = SIMPLE_AGENT_FLAVORS
ARM_FLAVORS = WORKFLOW_AGENT_FLAVORS
ALL_FLAVORS = AGENT_FLAVORS

AGENT_NAME = "swebench_agent"
AGENT_ROLE = (
    "Work in the local repository. Use bash for inspection, edits, "
    "and focused tests, then return a concise final note."
)
AGENT_SYSTEM_PROMPT = (
    "You are a software engineer interacting with a repository "
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
    """SWE-bench agent config; flavor from ``AGENT_FLAVOR``.

    Used by the generic ``agent_spec`` path for simple flavors; workflow arms
    are built by ``build_agent`` and never reach here. Capability-specific prompt
    addenda are owned by the generic runner / agent layer."""

    flavor = flavor_from_env()
    return AgentSpec(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        flavor=flavor,
    )


def chain_agent_spec(*, config: Mapping[str, Any]) -> AgentSpec:
    """SWE-bench agent config for the generic eval-chain runner."""

    return AgentSpec(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        flavor=_chain_agent_flavor(config),
    )


def chain_start_state(*, config: Mapping[str, Any], agent_name: str):
    """Create the SWE-bench-specific seed state for one repo chain."""

    display_name = _chain_display_name(config)
    task = (
        f"Repo chain for {display_name}. Solve instances for "
        "this repository in commit-time order. Carry useful context across "
        "tasks, but each instance's patch must address only the current problem."
    )
    return start_chain_state(
        task,
        agent_name=agent_name,
        metadata={
            "repo": str(config.get("repo") or ""),
            "chain_id": str(config.get("chain_id") or ""),
            "part_index": int(config.get("part_index", 1) or 1),
            "part_count": int(config.get("part_count", 1) or 1),
        },
    )


def chain_state_metadata(
    *, instance: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """SWE-bench metadata stored beside the generic chain state."""

    return {
        "repo": str(config.get("repo") or instance.get("repo") or ""),
        "chain_id": str(config.get("chain_id") or ""),
    }


def chain_task_details(
    *, instance: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach SWE-bench identity while the generic runner adds chain identity."""

    del config
    return {
        "swebench": {"instance_id": str(instance.get("instance_id") or "")},
    }


def chain_context_policy(
    *,
    provider: Provider,
    request_extra: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> ContextPolicy:
    """Build SWE-bench's compression policy for chain continuation."""

    if _chain_compression_strategy(config) != "summarize":
        return ContextPolicy()
    runtime = _runtime_config(config)
    compressor = make_llm_agent(
        name="swebench_compressor",
        provider=provider,
        role=(
            "Summarize older repo-chain context. Preserve durable "
            "facts, decisions, tool results, constraints, file paths, test "
            "signals, and unresolved questions. Omit low-value wording."
        ),
        request_extra=request_extra,
    )
    return ContextPolicy(
        strategy=SummarizeStrategy(
            compressor=compressor,
            threshold_tokens=int(runtime.get("threshold_tokens", 217600) or 217600),
            keep_recent=int(runtime.get("keep_recent", 4) or 4),
            preserve_kinds=tuple(
                runtime.get("preserve_kinds") or ("task", "system", "context")
            ),
        )
    )


def chain_result_metadata(
    *,
    instance: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """SWE-bench fields folded into the generic chain result."""

    chain_id = str(config.get("chain_id") or "")
    return {
        "repo": str(config.get("repo") or instance.get("repo") or ""),
        "chain_id": chain_id,
        "chain_part_index": int(config.get("part_index", 1) or 1),
        "chain_part_count": int(config.get("part_count", 1) or 1),
        "baseline_commit": str(context.get("baseline_commit") or ""),
    }


def chain_trace_metadata(
    *,
    instance: Mapping[str, Any],
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """SWE-bench fields added to chain trajectory metadata."""

    del instance
    return {
        "repo": str(result.get("repo") or config.get("repo") or ""),
        "chain_id": str(result.get("chain_id") or config.get("chain_id") or ""),
    }


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
        "Solve this repository task.",
        "",
        "## Environment",
        "- You are running inside the repository container.",
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


def apply_oracle(workspace: Path, instance: Mapping[str, Any]) -> None:
    """Apply the gold solution patch — the reference ("oracle") solution.

    Used by the framework's oracle run mode (no model) to validate that the
    suite is wired correctly: after this, `extract_result` should reproduce the
    gold patch. Applies only the solution `patch`, never `test_patch` (the agent
    is never asked to write tests). Raises on a missing/failed patch so a broken
    oracle instance fails loudly instead of silently producing an empty diff.
    """

    workspace = Path(workspace)
    patch_text = str(instance.get("patch") or "")
    if not patch_text.strip():
        raise ValueError("oracle run needs a non-empty 'patch' (gold) field")
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=workspace,
        input=patch_text,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"oracle git apply failed in {workspace}: {result.stderr.strip()}"
        )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the staged `git diff` as the SWE-bench prediction patch.

    For an arm flavor, the generic runner folds in the facade's per-step
    workflow breakdown after this returns (the facade stashes it on the run
    state), so this only has to produce the prediction patch.
    """

    context = context or {}
    record = dict(instance)
    language = str(context.get("language") or instance_language(record))
    commit = context.get("baseline_commit") or instance_base_commit(record)
    return {"model_patch": git_diff(Path(workspace), language=language, commit=commit)}


def evaluate(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score in the run environment: run the *official* eval script (ADR collapse-scorer-seam-into-run-primitive).

    The host staged the official eval script (generated from ``make_test_spec``)
    under EVAL_KEY; the generic runner threads it in as ``context["eval"]`` and
    only calls this hook when that gold is present. We run that exact script
    against the workspace the agent already edited — no fresh container — and
    capture its combined log into ``result.json`` (``eval_log``). Grading is done
    host-side by `evaluate_predictions.reuse_eval_row` via the official
    ``swebench`` grader (which needs the full test spec), so the verdict is
    parity-grade; capturing the official log here is what makes that grading
    trustable. Returns ``resolved: False`` with a diagnostic ``status`` when no
    eval script was staged, rather than crashing the run.
    """

    eval_script = str((context or {}).get("eval", {}).get("eval_script") or "")
    if not eval_script.strip():
        return {"resolved": False, "status": "no_eval_inputs"}

    script_path = Path(workspace) / "_sal_eval.sh"
    text = eval_script if eval_script.endswith("\n") else eval_script + "\n"
    script_path.write_text(text, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["/bin/bash", str(script_path)],
            cwd=str(workspace),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        script_path.unlink(missing_ok=True)
    return {
        "status": "eval_script_ran",
        "eval_log": (proc.stdout or "") + (proc.stderr or ""),
        "eval_exit_code": proc.returncode,
    }


def _repo_language() -> str:
    return config.REPO_LANGUAGE.get()


def _prepare_workflow_workspace(workdir: Path) -> None:
    """Suite-specific cleanup before workflow worktrees fork from the workspace."""

    update_info_exclude(workdir, language=_repo_language())


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    trace_put: ArtifactPut | None = None,
) -> Agent | None:
    """Build the arm facade, or None for a simple flavor.

    Returns ``None`` for ``bash`` / ``bash_task_read`` / ``bash_skills`` so the
    generic runner falls through to the ``agent_spec`` path (which keeps memory
    hooks). For an arm (``loop`` / ``goal`` / ``pdr``) it returns a facade ``Agent``
    whose single ``generate`` runs the whole arm on the task, leaves edits in the
    workspace (the prediction `extract_result` reads), and returns a short final
    message. The arm's per-step breakdown and sub-traces are owned by the facade
    (it stashes the breakdown on the run state for the generic runner to fold);
    this hook only injects the suite-specific worktree prep + trace sink.
    """

    flavor = flavor_from_env()
    if flavor not in ARM_FLAVORS:
        return None

    return build_flavor_agent(
        flavor=flavor,
        provider=provider,
        cwd=Path(cwd),
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        request_extra=request_extra,
        prepare_workspace=_prepare_workflow_workspace,
        trace_put=trace_put,
    )


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


def _runtime_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("config")
    return value if isinstance(value, Mapping) else {}


def _chain_agent_flavor(config: Mapping[str, Any]) -> str:
    return str(_runtime_config(config).get("agent_flavor") or flavor_from_env())


def _chain_compression_strategy(config: Mapping[str, Any]) -> str:
    return str(_runtime_config(config).get("compression_strategy") or "summarize")


def _chain_display_name(config: Mapping[str, Any]) -> str:
    display = str(config.get("chain_display_name") or "")
    if display:
        return display
    repo = str(config.get("repo") or config.get("chain_id") or "unknown")
    part_index = int(config.get("part_index", 1) or 1)
    part_count = int(config.get("part_count", 1) or 1)
    if part_count <= 1:
        return repo
    return f"{repo} part {part_index}/{part_count}"
