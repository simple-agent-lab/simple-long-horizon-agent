"""Pluggable completion checks for the goal loop.

Three independent ways to decide "is the objective verifiably done?":
- `model_declared_check` — the agent calls `update_goal` (terminal tool) to
  self-declare a status; we read it back from the recorded tool result.
- `command_verifier_check` — an objective command (e.g. `pytest && ruff check`)
  must exit 0.
- `judge_agent_check` — an independent agent returns `{"done", "reason"}` JSON.
- `default_check` — model-declared, GATED BY an optional verifier veto.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
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
        return text_result(
            f"goal {status}: {reason}",
            details={"goal_status": status, "reason": reason},
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


def command_verifier_check(command: str, *, cwd: str | Path | None = None) -> CompletionCheck:
    """Done iff `command` exits 0 (e.g. `pytest -q && ruff check`)."""

    def check(state: State) -> CompletionResult:
        result = run_bash(command, cwd=cwd)
        if result.exit_code == 0:
            return CompletionResult(done=True)
        return CompletionResult(done=False, reason=f"verifier exit {result.exit_code}")

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
    # Try direct parse first
    try:
        return dict(json.loads(output.strip()))
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to extract the first {...} object from prose
    match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
    if match:
        try:
            return dict(json.loads(match.group()))
        except (json.JSONDecodeError, ValueError):
            pass
    return {"done": False, "reason": "parse failure"}


def judge_agent_check(judge: Agent, objective: str) -> CompletionCheck:
    """Done per an independent judge agent returning `{"done", "reason"}` JSON."""

    def check(state: State) -> CompletionResult:
        transcript = final_output(state, state.messages[0].target if state.messages else "")
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
