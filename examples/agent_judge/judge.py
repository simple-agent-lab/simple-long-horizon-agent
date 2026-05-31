"""Judge agent — the container half of an "agent as judge" suite.

This is the whole point: the judge is *another agent run*. It gets a bash tool,
so it can investigate the candidate's workspace (run tests, grep, check the
answer is grounded) — that is what distinguishes agent-as-judge from a one-shot
LLM score. Its input is the candidate's product + a summary of the candidate's
*process* (number of model steps), wired in by the host from the shared store.

The verdict follows the same "product lives in the workspace" rule as any other
suite: the judge writes its verdict JSON to the workspace, and `extract_result`
reads it. The fallback rubric keeps this runnable under the deterministic fake
model (which does not actually write the file); with a real judge model the
verdict comes from the model's own investigation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec

VERDICT_FILE = "_judge_verdict.json"


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="judge",
        role="Assess the candidate answer against the rubric.",
        system_prompt=(
            "You are a strict evaluator. Investigate the workspace with bash to "
            "verify the candidate's claim is grounded, then WRITE your verdict as "
            f"JSON to ./{VERDICT_FILE} with keys: verdict ('pass'|'fail'), "
            "score (0..1), rationale."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    candidate = instance.get("candidate_result", {})
    return (
        f"Problem: {instance['problem']}\n"
        f"Candidate answer: {candidate.get('answer')!r}\n"
        f"Candidate took {instance.get('candidate_steps')} model step(s).\n"
        "Rubric: pass only if the answer is correct and grounded in the workspace.\n"
        f"Investigate with bash, then write your verdict JSON to ./{VERDICT_FILE}."
    )


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace)
    verdict_file = workspace / VERDICT_FILE
    if verdict_file.exists():
        # A real judge model investigated and wrote this via bash.
        verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
    else:
        # Deterministic fallback so the demo runs under the fake model.
        answer = (instance.get("candidate_result") or {}).get("answer")
        verdict = {
            "verdict": "pass" if answer else "fail",
            "score": 1.0 if answer else 0.0,
            "rationale": "fallback rubric: candidate produced a non-empty answer",
        }
    verdict.setdefault("criteria", {"grounded": verdict.get("score", 0.0)})
    verdict["judge_model"] = "fake-model"
    return {"judgment": verdict}
