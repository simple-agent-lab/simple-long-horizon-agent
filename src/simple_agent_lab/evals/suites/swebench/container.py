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
- `memory_artifacts(workspace, instance, *, context)` — optional: the raw
  products to keep in filesystem memory. Returns the collected solution patch as
  ``model_patch.diff`` so a memory chain carries how each issue was actually
  solved (only wired for the memory-chain experiment; no-op without memory).
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
and returns control to the generic outer loop. ``pdr`` fans out concurrent
attempts that each edit disk; if they shared one checkout they'd clobber each
other and the ``git diff`` would be garbage, so every attempt gets its own
``git worktree`` (off the baseline commit, outside the workspace) and resets it
to baseline in an ``init_state`` hook — keeping rounds independent, seeded only
by the brief.

It imports only the standard library and the installed wheel (`agents`,
`evals.*`, `workflow`, `trace`, and the sibling `patch` module), so it works
inside any SWE-bench image with no copied files.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from simple_agent_lab.state import State
from simple_agent_lab.tools.bash import DEFAULT_SUBMISSION_MARKER

from .patch import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
    update_info_exclude,
)

if TYPE_CHECKING:
    from simple_agent_lab.memory import FilesystemArtifact

# Back-compatible flavor-name aliases for older scripts/tests. The source of
# truth lives in `simple_agent_lab.agent_flavors`.
SIMPLE_FLAVORS = SIMPLE_AGENT_FLAVORS
ARM_FLAVORS = WORKFLOW_AGENT_FLAVORS
ALL_FLAVORS = AGENT_FLAVORS

AGENT_NAME = "swebench_agent"
AGENT_ROLE = (
    "Work in the local repository. Use bash for inspection, edits, "
    "and focused tests, then submit the patch with the required marker command."
)
AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer shell to "
    "solve programming tasks."
)


def agent_spec() -> AgentSpec:
    """SWE-bench agent config; flavor from ``AGENT_FLAVOR``.

    Used by the generic ``agent_spec`` path for simple flavors; workflow arms
    are built by ``build_agent`` and never reach here. Capability-specific prompt
    addenda are owned by the generic runner / agent layer."""

    _enable_swebench_runtime_defaults()
    flavor = flavor_from_env()
    return AgentSpec(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        flavor=flavor,
    )


def chain_agent_spec(*, config: Mapping[str, Any]) -> AgentSpec:
    """SWE-bench agent config for the generic eval-chain runner."""

    _enable_swebench_runtime_defaults()
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
    pr_description = [problem]
    if requirements:
        pr_description.extend(["", "## Requirements", requirements])
    if interface:
        pr_description.extend(["", "## Interface", interface])
    lines = [
        "<pr_description>",
        "Consider the following PR description:",
        "\n".join(pr_description),
        "</pr_description>",
        "",
        "<instructions>",
        "# Task Instructions",
        "",
        "## Overview",
        "",
        "You're a software engineer interacting continuously with a computer by "
        "submitting commands.",
        "You'll be helping implement necessary changes to meet requirements in "
        "the PR description.",
        "Your task is specifically to make changes to non-test files in the "
        "current directory in order to fix the issue described in the PR "
        "description in a way that is general and consistent with the codebase.",
        "<IMPORTANT>This is an interactive process where you will think and use "
        "AT LEAST ONE available tool, see the result, then think and choose "
        "your next tool call(s).</IMPORTANT>",
        "",
        "For each response:",
        "",
        "1. Include a THOUGHT section explaining your reasoning and what you're "
        "trying to accomplish",
        "2. Provide one or more bash tool calls to execute.",
        "",
        "## Important Boundaries",
        "",
        f"- MODIFY: Regular source code files in {workdir} (this is the working "
        "directory for all your subsequent commands)",
        "- DO NOT MODIFY: Tests, lockfiles (package-lock.json, yarn.lock, "
        "pnpm-lock.yaml, npm-shrinkwrap.json), or project metadata "
        "(pyproject.toml, .gitignore, setup.cfg, etc.)",
        "- Keep temporary reproduction helpers out of the final diff; write them "
        "under `/tmp/` or delete them before you stop.",
        "",
        "## Recommended Workflow",
        "",
        "1. Analyze the codebase by finding and reading relevant files",
        "2. Create a script to reproduce the issue",
        "3. Edit the source code to resolve the issue",
        "4. Verify your fix works by running your script again",
        "5. Test edge cases to ensure your fix is robust",
        "",
        "## Command Execution Rules",
        "",
        "You are operating in an environment where",
        "",
        "1. You issue at least one tool call",
        "2. The system executes shell commands from `bash` calls in a subshell",
        "3. You see the result(s)",
        "4. You write your next tool call(s)",
        "",
        "Each response should include:",
        "",
        "1. **THOUGHT** text where you explain your analysis and plan",
        "2. At least one bash tool call for the next useful action",
        "",
        "**CRITICAL REQUIREMENTS:**",
        "",
        "- Your response SHOULD include THOUGHT text explaining what you're doing",
        "- Your response MUST include AT LEAST ONE bash tool call. Do not add a "
        "dummy `bash` call solely to satisfy this rule.",
        "- You can make MULTIPLE tool calls in a single response when the actions "
        "are independent (e.g., reading different parts of the codebase).",
        "- Directory or environment variable changes are not persistent. Every "
        "action is executed in a new subshell.",
        "- However, you can prefix any action with "
        "`MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load "
        "environment variables from files.",
        "- A full Linux shell is available; install missing tools only if "
        "strictly needed.",
        "- Always use non-interactive flags (`-y`, `-f`, `--no-pager`) for "
        "commands. Avoid interactive tools like `vi`, `nano`, or any command "
        "that requires user input.",
        "- Keep command output focused. If output may be long, use selective "
        "commands such as `head`, `tail`, `sed -n`, or redirect output to a "
        "file and inspect only the relevant slices.",
        "",
        "Example of a CORRECT response:",
        "<example_response>",
        "THOUGHT: I need to understand the relevant code before editing. I will "
        "find likely files and inspect the nearby implementation.",
        "",
        "[Makes one or more bash tool calls to inspect the repository.]",
        "</example_response>",
        "",
        "## Submission",
        "",
        "When you've completed your work, you MUST submit your changes as a git "
        "patch.",
        "Follow these steps IN ORDER, with SEPARATE commands:",
        "",
        "Step 1: Create the patch file",
        "Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only "
        "the source files you modified.",
        "Do NOT commit your changes.",
        "",
        "<IMPORTANT>",
        "The patch must only contain changes to the specific source files you "
        "modified to fix the issue.",
        "Do not submit file creations or changes to any of the following files:",
        "",
        "- test and reproduction files",
        "- helper scripts, tests, or tools that you created",
        "- lockfiles (package-lock.json, yarn.lock, pnpm-lock.yaml, "
        "npm-shrinkwrap.json)",
        "- installation, build, packaging, configuration, or setup scripts "
        "(pyproject.toml, setup.cfg, etc.) unless they are directly part of the "
        "issue you were fixing",
        "- binary or compiled files",
        "</IMPORTANT>",
        "",
        "Step 2: Verify your patch",
        "Inspect patch.txt to confirm it only contains your intended changes and "
        "headers show `--- a/` and `+++ b/` paths.",
        "",
        "Step 3: Submit (EXACT command required)",
        "You MUST use this EXACT command to submit:",
        "",
        "```bash",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        "```",
        "",
        "If the command fails (nonzero exit status), it will not submit.",
        "",
        "<CRITICAL>",
        "- Creating/viewing the patch and submitting it MUST be separate commands "
        "(not combined with &&).",
        "- If you modify patch.txt after verifying, you SHOULD verify again "
        "before submitting.",
        "- You CANNOT continue working (reading, editing, testing) in any way on "
        "this task after submitting.",
        "</CRITICAL>",
        "</instructions>",
    ]
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


def _collected_patch(
    workspace: Path, instance: Mapping[str, Any], context: Mapping[str, Any]
) -> str:
    """The workspace git diff scored as ``model_patch`` (language-aware ignores).

    Shared by ``extract_result`` and ``memory_artifacts`` so the patch stored in
    filesystem memory is exactly the patch used for scoring.
    """

    record = dict(instance)
    language = str(context.get("language") or instance_language(record))
    commit = context.get("baseline_commit") or instance_base_commit(record)
    return git_diff(Path(workspace), language=language, commit=commit)


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    state: State | None = None,
) -> dict[str, Any]:
    """Collect both SWE-bench patch products.

    ``model_patch`` remains the original Simple Agent Lab collected workspace
    diff for backward-compatible scoring. ``model_submitted_patch`` is the
    model-authored ``patch.txt`` / submission output, matching mini-SWE-agent's
    explicit-patch protocol.

    For an arm flavor, the generic runner folds in the facade's per-step
    workflow breakdown after this returns (the facade stashes it on the run
    state), so this only has to produce the prediction patch.
    """

    context = context or {}
    collected = _collected_patch(Path(workspace), instance, context)
    submitted, submitted_source = _submitted_patch(Path(workspace), state=state)
    return {
        "model_patch": collected,
        "model_patch_source": "collected_git_diff",
        "model_submitted_patch": submitted,
        "model_submitted_patch_source": submitted_source,
    }


def memory_artifacts(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[FilesystemArtifact, ...]:
    """Store the model's solution patch as filesystem-memory evidence.

    The generic runner calls this at memory ``finish()`` (SESSION_END), before
    ``extract_result`` runs, while the edited workspace still exists. It returns
    the collected workspace diff — the same ``model_patch`` used for scoring — as
    ``model_patch.diff`` so a later issue in the same memory chain can read how
    this one was actually solved, and so the run-end distiller sees the concrete
    change instead of only the transcript. An empty diff yields no artifact:
    there is nothing durable to keep, and a "no memory update" result is fine.
    """

    from simple_agent_lab.memory import FilesystemArtifact

    patch = _collected_patch(Path(workspace), instance, context or {})
    if not patch.strip():
        return ()
    return (
        FilesystemArtifact(
            name="model_patch.diff",
            content=patch,
            description="Collected workspace diff (the model's solution patch).",
        ),
    )


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
    workspace (the prediction `extract_result` reads), and returns control to the
    generic runner. The arm's per-step breakdown and sub-traces are owned by the
    facade (it stashes the breakdown on the run state for the generic runner to
    fold); this hook only injects the suite-specific worktree prep + trace sink.
    """

    _enable_swebench_runtime_defaults()
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


def _enable_swebench_runtime_defaults() -> None:
    """Match the long-running SWE-bench Pro baseline envelope by default."""

    # env-ok: suite-level defaults for registered EnvVar knobs.
    os.environ.setdefault(config.BASH_SUBMISSION_MARKER.name, DEFAULT_SUBMISSION_MARKER)
    # env-ok: suite-level defaults for registered EnvVar knobs.
    os.environ.setdefault(config.BASH_DEFAULT_TIMEOUT.name, "3000")
    # env-ok: suite-level defaults for registered EnvVar knobs.
    os.environ.setdefault(config.BASH_MAX_TIMEOUT.name, "3000")
    # env-ok: suite-level defaults for registered EnvVar knobs.
    os.environ.setdefault(config.BASH_MAX_OUTPUT_CHARS.name, "10000")
    # env-ok: suite-level defaults for registered EnvVar knobs.
    os.environ.setdefault(config.LLM_REQUEST_TIMEOUT.name, "1800")


def _submitted_patch(workspace: Path, *, state: State | None) -> tuple[str, str]:
    """Return the model-authored patch, preferring the captured submit output."""

    from_state = _submitted_patch_from_state(state)
    if from_state:
        return _normalize_patch(from_state), "tool_submission"
    patch_txt = workspace / "patch.txt"
    if patch_txt.is_file():
        try:
            return _normalize_patch(patch_txt.read_text(encoding="utf-8")), "patch_txt"
        except OSError:
            return "", ""
    return "", ""


def _submitted_patch_from_state(state: State | None) -> str:
    if state is None:
        return ""
    for message in reversed(state.messages):
        details = message.sidecar.get("details")
        if not isinstance(details, Mapping):
            continue
        for value in reversed(list(details.values())):
            if not isinstance(value, Mapping):
                continue
            submission = value.get("submission")
            if isinstance(submission, str) and submission.strip():
                return submission
            raw_stdout = value.get("raw_stdout")
            if isinstance(raw_stdout, str):
                parsed = _submission_after_marker(raw_stdout)
                if parsed.strip():
                    return parsed
    return ""


def _submission_after_marker(output: str) -> str:
    lines = output.lstrip().splitlines(keepends=True)
    if not lines or lines[0].strip() != DEFAULT_SUBMISSION_MARKER:
        return ""
    return "".join(lines[1:])


def _normalize_patch(patch: str) -> str:
    stripped = patch.strip()
    return stripped + ("\n" if stripped else "")


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
