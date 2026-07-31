"""Judge agent — the container half of an "agent as judge" suite.

This is the whole point: the judge is *another agent run*. It gets a bash tool,
so it can investigate the candidate's workspace (run tests, grep, check the
answer is grounded) — that is what distinguishes agent-as-judge from a one-shot
LLM score. Its input is the candidate's product + a summary of the candidate's
*process* (number of model steps), wired in by the host from the shared store.

The verdict follows the same "product lives in the workspace" rule as any other
suite: the judge writes its verdict JSON to the workspace, and `extract_result`
reads it. If the model never writes the file, a safety-net fallback defers to the
candidate's own in-env verdict rather than inventing a score.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_long_horizon_agent.evals.protocols import AgentSpec, LaunchSpec

VERDICT_FILE = "_judge_verdict.json"


class JudgeSuite:
    """Host half for the judge run. Passes the candidate context straight through.

    `launch_spec` declares the container shape a container backend would use;
    `LocalProcessBackend` supplies its own workspace and ignores it."""

    name = "judge"
    container_module = "examples.bench_suite.judge"
    image = "python:3.12-slim"
    workdir = "/workspace"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image=self.image, workdir=self.workdir)

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        # Keep candidate_result / candidate_steps — the judge needs them.
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        # No in-environment scoring: the judge run *is* the scoring step, and its
        # verdict is this run's own result.json (read back by the host).
        return None


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
        f"Candidate reported output: {candidate.get('answer')!r}\n"
        f"Candidate took {instance.get('candidate_steps')} model step(s).\n"
        "Rubric: pass only if app.py is genuinely fixed and running it prints 42 "
        "(run it yourself to confirm — do not trust the reported output).\n"
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
        # Safety-net fallback if the model never wrote the verdict file: defer to
        # the candidate's own in-env grade rather than inventing one.
        passed = bool((instance.get("candidate_result") or {}).get("passed"))
        verdict = {
            "verdict": "pass" if passed else "fail",
            "score": 1.0 if passed else 0.0,
            "rationale": "fallback: judge file missing; deferred to in-env verdict",
        }
    verdict.setdefault("criteria", {"grounded": verdict.get("score", 0.0)})
    return {"judgment": verdict}
