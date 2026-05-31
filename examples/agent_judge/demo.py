"""Agent-as-judge demo on the *current* framework — same code, swappable backend.

A "pipeline" here is just plain Python over the shared `ArtifactStore`:

    candidate run  --put-->  ArtifactStore  --get-->  judge run

Both are ordinary `run_suite_instance(...)` calls. This demo uses
`LocalProcessBackend` (in-process, no Docker, fake provider) so it runs anywhere
and iterates fast. To run the *same* suites containerized — locally or across
machines — swap one argument:

    backend = LocalProcessBackend(workspace=ws)   # local development
    backend = LocalDockerBackend()                # one machine, in a container
    backend = RemoteDockerBackend(host="...")     # multi-machine (future)

Nothing else changes: the candidate/judge container halves run identically.

Run (from the repo root):  uv run python -m examples.agent_judge.demo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from examples.agent_judge.candidate import CandidateSuite
from examples.agent_judge.judge import JudgeSuite
from simple_agent_lab.evals import (
    RESULT_KEY,
    TRACE_KEY,
    LocalDirStore,
    LocalProcessBackend,
    run_suite_instance,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = LocalDirStore(root)

        # A tiny workspace the candidate (and later the judge) can inspect.
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
        (workspace / "README.md").write_text("# demo\n", encoding="utf-8")

        # In-process backend: runs the agent loop here, no Docker. The workspace
        # stands in for what a container image would provide.
        backend = LocalProcessBackend(workspace=workspace)

        # ---- 1. candidate run -------------------------------------------------
        task = {"instance_id": "demo-1", "problem": "Summarize the workspace."}
        cand = run_suite_instance(
            suite=CandidateSuite(),
            instance=task,
            backend=backend,
            store=store,
            run_root=root,
            run_id="cand",
            provider="fake",
        )
        cand_store = store.bind(cand.run_dir)
        candidate = json.loads(cand_store.get(RESULT_KEY).decode("utf-8"))
        cand_trace = json.loads(cand_store.get(TRACE_KEY).decode("utf-8"))

        # ---- 2. host glue (the "pipeline"): candidate artifacts -> judge input
        judge_instance = {
            "instance_id": "demo-1",
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
            provider="fake",
        )
        judged = json.loads(
            store.bind(judged_run.run_dir).get(RESULT_KEY).decode("utf-8")
        )

        print("candidate result :", json.dumps(candidate, ensure_ascii=False))
        print(
            "judgment :", json.dumps(judged["judgment"], ensure_ascii=False, indent=2)
        )
        print("\nstore tree:")
        for p in sorted(root.rglob("*.json*")):
            print("  ", p.relative_to(root))


if __name__ == "__main__":
    main()
