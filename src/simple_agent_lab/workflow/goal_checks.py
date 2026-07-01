"""Pluggable completion checks for the goal loop.

Ways to decide "is the objective verifiably done?":
- `model_declared_check` — the agent calls `update_goal` (terminal tool) to
  self-declare a status; we read it back from the recorded tool result.
- `command_verifier_check` — a fixed command (e.g. `pytest && ruff check`) must
  exit 0.
- `judge_agent_check` — an independent agent returns `{"done", "reason"}` JSON.
- `default_check` — model-declared, GATED BY an optional verifier veto.
- `verified_completion_check` — model-declared, GATED BY an LLM judge reading the
  transcript (the judge sees only the agent's narrative, so it is noisy).
- `executed_completion_check` — model-declared, GATED BY re-running the model's
  own declared `verify_command` (exit 0). A real execution rather than a second
  model, so it can't be fooled by a convincing-but-wrong summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider, extract_json_object
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import tool_results_of
from simple_agent_lab.state import State
from simple_agent_lab.tools import AgentTool, ToolResult, ToolUpdateFn, text_result
from simple_agent_lab.tools.bash import run_bash

from .base import final_output, run_agent
from .goal_loop import CompletionCheck, CompletionResult

UPDATE_GOAL_TOOL_NAME = "update_goal"


def update_goal_tool(*, name: str = UPDATE_GOAL_TOOL_NAME) -> AgentTool:
    """A terminal tool the model calls to declare a goal status.

    Returns `ToolResult(terminate=True, details={"goal_status": ...})` — the
    existing `tool_terminate` seam (`core.py:295-300`). The status + reason are
    carried in `details`, which the loop bundles into the recorded tool_result
    message's sidecar (keyed by tool_call_id) for `model_declared_check` to read.
    """

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: Any,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        status = str(args.get("status", "")).strip() or "complete"
        reason = str(args.get("reason", "")).strip()
        verify_command = str(args.get("verify_command", "")).strip()
        return text_result(
            f"goal {status}: {reason}",
            details={
                "goal_status": status,
                "reason": reason,
                "verify_command": verify_command,
            },
            terminate=True,
        )

    return AgentTool(
        name=name,
        description="Declare the goal's terminal status once you have verified it.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["complete", "blocked"]},
                "reason": {"type": "string"},
                "verify_command": {
                    "type": "string",
                    "description": (
                        "When status=complete, the exact shell command that proves "
                        "the objective (e.g. the test/repro command you ran). A "
                        "command-gated check RE-RUNS it and rejects completion "
                        "unless it exits 0, so it must be a real verification, not "
                        "a trivially-passing command."
                    ),
                },
            },
            "required": ["status"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _last_goal_details(state: State) -> dict[str, Any] | None:
    """Most recent `update_goal` details from the recorded tool_result bundle."""
    for message in reversed(state.messages):
        if message.kind != "tool_result":
            continue
        details = (message.sidecar or {}).get("details") or {}
        for block in tool_results_of(message.content):
            payload = details.get(block.tool_call_id)
            if isinstance(payload, dict) and "goal_status" in payload:
                return payload
    return None


def model_declared_check(state: State) -> CompletionResult:
    """Done iff the agent called `update_goal` with status `complete`."""
    payload = _last_goal_details(state)
    if payload is None:
        return CompletionResult(done=False)
    status = payload.get("goal_status")
    reason = payload.get("reason", "")
    if status == "complete":
        return CompletionResult(done=True, reason=reason)
    if status == "blocked":
        return CompletionResult(done=False, blocked=True, reason=reason)
    return CompletionResult(done=False)


def command_verifier_check(
    command: str, *, cwd: str | Path | None = None, exec_prefix: tuple[str, ...] = ()
) -> CompletionCheck:
    """Done iff `command` exits 0 (e.g. `pytest -q && ruff check`)."""

    def check(state: State) -> CompletionResult:
        result = run_bash(command, cwd=cwd, exec_prefix=exec_prefix)
        if result.exit_code == 0:
            return CompletionResult(done=True)
        return CompletionResult(done=False, reason=f"verifier exit {result.exit_code}")

    return check


def executed_completion_check(
    *, cwd: str | Path | None = None, exec_prefix: tuple[str, ...] = ()
) -> CompletionCheck:
    """Model-declared completion GATED BY re-running the model's own verify command.

    The reusable, judge-free completion gate (the alternative to the LLM
    `verified_completion_check`): the agent calls `update_goal(status=complete,
    verify_command=...)` and this check RE-RUNS that exact command in `cwd`,
    accepting completion only if it exits 0. The gate is a real execution, not a
    second model reading a transcript, so a convincing-but-wrong claim can't pass
    (the broken fix's own test command fails) and a correct-but-tersely-summarized
    fix isn't penalized (the command still passes). `status=blocked` propagates
    as usual; a `complete` without a real `verify_command` is held open so the
    model is forced to ground completion in something runnable.
    """

    def check(state: State) -> CompletionResult:
        declared = model_declared_check(state)
        if not declared.done:
            return declared  # not-yet-complete, or blocked — pass through.
        payload = _last_goal_details(state) or {}
        command = str(payload.get("verify_command", "")).strip()
        if not command:
            # An empty command would `run_bash("")` to exit 0 and pass the gate
            # for free — hold completion open until a real command is supplied.
            return CompletionResult(
                done=False,
                reason="completion needs a verify_command to re-run",
            )
        result = run_bash(command, cwd=cwd, exec_prefix=exec_prefix)
        if result.exit_code == 0:
            return CompletionResult(done=True, reason=declared.reason)
        return CompletionResult(
            done=False,
            reason=f"verify_command re-run failed (exit {result.exit_code})",
        )

    return check


def _judge_prompt(objective: str, transcript: str) -> str:
    """Prompt for the judge agent: evaluate whether `objective` is verifiably done."""
    return (
        "You are an independent judge. Given the objective and the agent's work "
        "transcript, decide if the objective is verifiably complete.\n\n"
        f"Objective:\n{objective}\n\n"
        f"Agent transcript:\n{transcript}\n\n"
        'Respond with ONLY valid JSON: {"done": true/false, "reason": "..."}'
    )


def _parse_judge_json(output: str) -> dict[str, Any]:
    """Extract the first JSON object from `output`; fall back to done=false."""
    return extract_json_object(output) or {"done": False, "reason": "parse failure"}


def judge_agent_check(judge: Agent, objective: str) -> CompletionCheck:
    """Done per an independent judge agent returning `{"done", "reason"}` JSON."""

    def check(state: State) -> CompletionResult:
        transcript = final_output(
            state, state.messages[0].target if state.messages else ""
        )
        step = run_agent(judge, _judge_prompt(objective, transcript))
        verdict = _parse_judge_json(step.output)
        return CompletionResult(
            done=bool(verdict.get("done")),
            reason=str(verdict.get("reason", "")),
        )

    return check


def default_check(verifier: CompletionCheck | None = None) -> CompletionCheck:
    """Model-declared completion, GATED BY an optional verifier veto.

    The model must declare done; if a `verifier` is supplied, it can override a
    declared-done back to not-done (model says done, verifier vetoes).
    """

    def check(state: State) -> CompletionResult:
        declared = model_declared_check(state)
        if not declared.done:
            return declared
        if verifier is None:
            return declared
        verified = verifier(state)
        if verified.done:
            return CompletionResult(done=True)
        return CompletionResult(done=False, reason=f"verifier veto: {verified.reason}")

    return check


# --------------------------------------------------------------------------- #
# Verified completion: the reusable "judge gate" for `run_goal_loop`
#
# A task-agnostic packaging of the pattern that worked for agentic coding:
# the model declares done, an INDEPENDENT judge vetoes unless the transcript
# shows the objective was actually verified (the relevant checks/tests/commands
# were run and succeeded), and the loop keeps going otherwise. The prompts here
# are deliberately domain-neutral ("checks/tests/commands") so any verifiable
# task — not just SWE-bench — can reuse it. Callers may pass their own
# `system_prompt` to `make_completion_judge` to add domain flavor.
# --------------------------------------------------------------------------- #
COMPLETION_JUDGE_ROLE = "Independently verify an objective is really complete."
COMPLETION_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, independent reviewer. You are given an objective and the "
    "agent's work transcript. Answer done=true ONLY if the transcript shows the "
    "objective was actually accomplished AND verified with concrete evidence — "
    "the relevant checks/tests/commands were run and their output shows success "
    "— not merely a claim of completion, a plan, or a 'looks correct' review. If "
    "there is no concrete evidence of executed, successful verification, answer "
    'done=false. Respond with ONLY valid JSON: {"done": true/false, "reason": "..."}.'
)

# System-prompt addendum for the worked agent. A "completion audit" in the spirit
# of a rigorous self-verification: completion is unproven until proven, against
# the actual current state, per requirement — and the declared completion must
# carry the exact command a command-gated check re-runs.
VERIFY_BEFORE_DONE_ADDENDUM = (
    "Completion is UNPROVEN until you prove it. Before calling `update_goal` with "
    "status=complete, run a completion audit:\n"
    "1. Re-derive the objective's concrete requirements — every explicit item, "
    "named file/function, test, command, and behavior it asks for. Preserve the "
    "original scope; do not redefine success around the easy part.\n"
    "2. For EACH requirement, gather AUTHORITATIVE evidence by running a concrete "
    "check (test results / command output) against the current files, not your "
    "memory of what you changed. Treat uncertain, indirect, or 'looks correct' "
    "evidence as NOT done.\n"
    "3. The audit must PROVE completion, not merely fail to find obvious remaining "
    "work.\n"
    "Only when the audit proves every requirement, call `update_goal` with "
    "status=complete AND set `verify_command` to the exact shell command that "
    "proves the fix (the test/repro command you ran). The harness RE-RUNS that "
    "command and rejects completion unless it exits 0, so it must be a real "
    "verification. If the SAME blocker recurs across turns, call `update_goal` "
    "with status=blocked and the reason. Do not declare complete merely because "
    "you are stopping or the budget is low."
)

# Per-continuation nudge (no untrusted-data wrapper — for a trusted objective the
# agent already saw verbatim on turn 1). Pass as `continuation_prompt` to
# `run_goal_loop`.
VERIFY_CONTINUATION = (
    "Do not assume the objective is done. Re-derive its concrete requirements and, "
    "for EACH, verify against current-state evidence by re-running the relevant "
    "checks/tests from the actual files (not memory). Treat weak or indirect "
    "evidence as not done — the audit must prove completion, not merely fail to "
    "find remaining work. If everything genuinely passes, call `update_goal` with "
    "status=complete and set `verify_command` to the exact command that proves it "
    "(the harness re-runs it and requires exit 0). If the SAME blocker keeps "
    "recurring, call `update_goal` with status=blocked. Otherwise keep working."
)


def make_completion_judge(
    provider: Provider,
    *,
    name: str = "completion_judge",
    role: str = COMPLETION_JUDGE_ROLE,
    system_prompt: str = COMPLETION_JUDGE_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the tool-free independent completion judge for `verified_completion_check`."""

    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=(),
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )


def verified_completion_check(judge: Agent, objective: str) -> CompletionCheck:
    """Model-declared completion GATED BY an independent judge's evidence check.

    `default_check(verifier=judge_agent_check(judge, objective))`: the model must
    call `update_goal(status=complete)` AND the `judge` must agree the transcript
    shows real, executed verification — otherwise the loop continues. This is the
    general, reusable "judge gate" optimization for `run_goal_loop`.
    """

    return default_check(verifier=judge_agent_check(judge, objective))
