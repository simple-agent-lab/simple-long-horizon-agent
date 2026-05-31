"""Agent-as-judge demo on the *current* framework — no new abstraction.

A "pipeline" here is just plain Python over the shared `ArtifactStore`:

    candidate run  --put-->  ArtifactStore  --get-->  judge run

Both runs are ordinary agent runs through the generic in-container runner
(`run_in_container`). This demo calls it in-process (no Docker, fake provider) so
it runs anywhere; under Docker you would wrap each call in
`run_suite_instance(suite=..., backend=LocalDockerBackend(), store=...)` and the
suite's container half (these `candidate` / `judge` modules) would run inside the
image unchanged.

Run (from the repo root):  uv run python -m examples.agent_judge.demo

Running as a module (`-m`) puts the repo root on ``sys.path`` so the dotted
container-module paths resolve, and ``uv run`` provides the installed
``simple_agent_lab`` — so no ``sys.path`` juggling is needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from simple_agent_lab.evals import INSTANCE_KEY, TRACE_KEY, LocalDirStore
from simple_agent_lab.evals.in_container import run_in_container
from simple_agent_lab.llm import Provider

FAKE = Provider(id="fake", api="fake", model="fake-model")


def _run(*, module: str, instance: dict, store, workdir: Path, suite: str) -> dict:
    """One agent run: seed the instance, drive the loop, return the product."""

    store.put(INSTANCE_KEY, (json.dumps(instance) + "\n").encode("utf-8"))
    result, _state = run_in_container(
        instance=instance,
        container_module=module,
        provider=FAKE,
        workdir=workdir,
        max_turns=4,
        store=store,
        trace_id=f"{suite}.{instance['instance_id']}",
        producer=f"suite:{suite}",
        suite_name=suite,
    )
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = LocalDirStore(root)

        # A tiny workspace the candidate (and later the judge) can inspect.
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
        (workspace / "README.md").write_text("# demo\n", encoding="utf-8")

        # ---- 1. candidate run -------------------------------------------------
        task = {"instance_id": "demo-1", "problem": "Summarize the workspace."}
        cand_store = store.bind(root / "cand")
        candidate = _run(
            module="examples.agent_judge.candidate",
            instance=task,
            store=cand_store,
            workdir=workspace,
            suite="candidate",
        )
        cand_trace = json.loads(cand_store.get(TRACE_KEY).decode("utf-8"))

        # ---- 2. host glue (the "pipeline"): candidate artifacts -> judge input
        judge_instance = {
            "instance_id": "demo-1",
            "problem": task["problem"],
            "candidate_result": candidate,  # the product
            "candidate_steps": len(cand_trace.get("model_turns", [])),  # the process
        }

        # ---- 3. judge run (agent-as-judge over the same workspace) ------------
        judge_store = store.bind(root / "judge")
        judged = _run(
            module="examples.agent_judge.judge",
            instance=judge_instance,
            store=judge_store,
            workdir=workspace,
            suite="judge",
        )

        print("candidate result :", json.dumps(candidate, ensure_ascii=False))
        print(
            "judgment :", json.dumps(judged["judgment"], ensure_ascii=False, indent=2)
        )
        # Both runs' artifacts (result.json + trajectory.jsonl) sit under the one
        # store, keyed by run — ready for a third step (aggregation, a panel of
        # judges, training-example export) with no extra framework.
        print("\nstore tree:")
        for p in sorted(root.rglob("*.json*")):
            print("  ", p.relative_to(root))


if __name__ == "__main__":
    main()
