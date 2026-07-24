"""Worked bench-suite demo on the *current* framework — a real model, real task.

It shows the two scoring paths a suite can take:

    run  --put-->  ArtifactStore  --get-->  { in-env verdict (result.json)
                                              judge run      (agent-as-judge) }

The candidate is a real agent that must *fix a bug* in `app.py`; it scores itself
*in the run environment* (its `evaluate` hook runs the fixed program and compares
the output to the gold, writing the verdict into `result.json`), so its score is
read straight back — no separate scoring phase. A second *agent* then judges the
same work. Both are plain Python over the shared `ArtifactStore`.

This demo uses `LocalProcessBackend` (in-process, no Docker) with a real
OpenAI-compatible model. Set the model env first:

    export OPENAI_MODEL=gpt-4o-mini
    export OPENAI_AUTH_TOKEN=sk-...
    # export OPENAI_BASE_URL=https://...   # optional, for a compatible gateway
    # export API_KIND=openai-chat          # or openai-responses

To run the *same* suites containerized — locally or across machines — swap one
argument (the candidate/judge container halves run identically):

    backend = LocalProcessBackend(workspace=ws)   # local development
    backend = LocalDockerBackend()                # one machine, in a container
    backend = RemoteDockerBackend(host="...")     # multi-machine (future)

Run (from the repo root):  uv run python -m examples.bench_suite.demo
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from examples.bench_suite.candidate import APP_FILE, ExampleBenchSuite
from examples.bench_suite.judge import JudgeSuite
from simple_agent_lab.evals import (
    RESULT_KEY,
    TRACE_KEY,
    LocalDirStore,
    LocalProcessBackend,
    run_dataset,
    run_suite_instance,
)

# A one-file program with a bug: `add` subtracts, so it prints -2 instead of 42.
# The candidate agent must find and fix it; we grade by running the fixed file.
BUGGY_APP = """\
def add(a, b):
    return a - b  # BUG: should add


if __name__ == "__main__":
    print(add(20, 22))
"""


def _provider_env() -> dict[str, str]:
    keys = ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN", "OPENAI_BASE_URL")
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


def main() -> None:
    missing = [
        k for k in ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN") if not os.environ.get(k)
    ]
    if missing:
        print(
            "This demo runs a real model. Set OPENAI_MODEL and OPENAI_AUTH_TOKEN "
            "(optionally OPENAI_BASE_URL / API_KIND) first.\n"
            f"Missing: {', '.join(missing)}"
        )
        return

    provider_env = _provider_env()
    api_kind = os.environ.get("API_KIND", "openai-chat")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = LocalDirStore(root)

        # The workspace the candidate fixes and the judge later inspects.
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / APP_FILE).write_text(BUGGY_APP, encoding="utf-8")

        # In-process backend: runs the agent loop here, no Docker. The workspace
        # stands in for what a container image would provide.
        backend = LocalProcessBackend(workspace=workspace)

        # ---- 1. candidate run (scores itself in the run environment) ----------
        # `expected` is the gold program output: `task_input` hides it from the
        # agent, `eval_inputs` stages it, and the suite's `evaluate` hook runs the
        # fixed program and grades it — so the verdict is already in result.json.
        task = {
            "instance_id": "fix-add",
            "problem": "Fix the bug in app.py so that running it prints exactly 42.",
            "expected": "42",
        }
        report = run_dataset(
            suite=ExampleBenchSuite(),
            instances=[task],
            backend=backend,
            store=store,
            run_root=root,
            run_id="cand",
            provider="openai",
            api_kind=api_kind,
            provider_env=provider_env,
            max_turns=25,
        )
        cand_result = report.results[0]
        if cand_result.artifacts is None:
            print("candidate run failed:", cand_result.error)
            return
        cand_store = store.bind(cand_result.artifacts.run_dir)
        candidate = json.loads(cand_store.get(RESULT_KEY).decode("utf-8"))
        cand_trace = json.loads(cand_store.get(TRACE_KEY).decode("utf-8"))

        # ---- 2. host glue (the "pipeline"): candidate artifacts -> judge input
        judge_instance = {
            "instance_id": "fix-add",
            "problem": task["problem"],
            "candidate_result": candidate,  # the product
            "candidate_steps": len(cand_trace.get("model_turns", [])),  # the process
        }

        # ---- 3. judge run (agent-as-judge over the same workspace) ------------
        judged_run = run_suite_instance(
            suite=JudgeSuite(),
            instance=judge_instance,
            backend=backend,
            store=store,
            run_root=root,
            run_id="judge",
            provider="openai",
            api_kind=api_kind,
            provider_env=provider_env,
            max_turns=25,
        )
        judged = json.loads(
            store.bind(judged_run.run_dir).get(RESULT_KEY).decode("utf-8")
        )

        verdict = {k: candidate[k] for k in ("passed", "score") if k in candidate}
        print("candidate result :", json.dumps(candidate, ensure_ascii=False))
        print("in-env verdict   :", json.dumps(verdict, ensure_ascii=False))
        print(
            "judgment :", json.dumps(judged["judgment"], ensure_ascii=False, indent=2)
        )
        print("\nstore tree:")
        for p in sorted(root.rglob("*.json*")):
            print("  ", p.relative_to(root))


if __name__ == "__main__":
    main()
