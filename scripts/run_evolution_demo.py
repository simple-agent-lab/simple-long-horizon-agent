"""Walk the evolution substrate end to end, deterministically (no model calls).

Demonstrates the loop from docs/design/20260610-evolution-framework-spec.md
with a stub rollout whose reward depends on the bundle's playbook content:
stage an initial bundle, propose a candidate, gate it, promote on acceptance,
then show the decision log and a rollback.

Usage:
    PYTHONPATH=src python scripts/run_evolution_demo.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from simple_agent_lab.evolution import (
    EvalSlice,
    Manifest,
    bundle_hash,
    gate,
    promote,
    read_decisions,
    resolve,
    stage_bundle,
)


def stub_rollout(runs_root: Path):
    """Reward 0.4, +0.3 when the playbook mentions checking pytest fixtures —
    a stand-in for a real eval suite so the walkthrough stays deterministic."""

    def rollout(bundle_dir: Path, run_id: str) -> list[Path]:
        playbook = bundle_dir / "playbook.md"
        text = playbook.read_text() if playbook.exists() else ""
        reward = 0.4 + (0.3 if "pytest fixtures" in text else 0.0)
        instance = runs_root / run_id / "demo-1"
        (instance / "out").mkdir(parents=True, exist_ok=True)
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [instance]

    return rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="evolution-demo-"))
    print(f"workspace: {workspace}\n")

    # 1. An initial task bundle becomes "current" by promotion.
    initial = stage_bundle(
        workspace,
        manifest=Manifest(level="task", producer="demo", note="initial"),
        edits={"prompt.md": "You are a careful software agent."},
    )
    promote(workspace, "task", initial)
    print(f"initial bundle promoted: {bundle_hash(initial)}")

    # 2. A candidate = file edits on top of the current bundle (what the
    #    evolution agent's write_candidate tool does).
    candidate = stage_bundle(
        workspace,
        base=resolve(workspace, "task"),
        manifest=Manifest(
            level="task",
            parent=bundle_hash(initial),
            producer="demo",
            evidence=["trace:demo-failure-7"],
            note="playbook bullet: check pytest fixtures before editing tests",
        ),
        edits={
            "playbook.md": "- When tests fail on setup, check pytest fixtures first.\n"
        },
    )
    print(f"candidate staged:        {bundle_hash(candidate)}")

    # 3. The gate: rollout both on the frozen slice, judge, log the decision.
    result = gate(
        workspace,
        baseline=resolve(workspace, "task"),
        candidate=candidate,
        slice_=EvalSlice(suite="demo", instances=({"instance_id": "demo-1"},)),
        rollout=stub_rollout(workspace / "runs"),
        kind="playbook",
        episode="ep-000001",
    )
    print(f"\ngate {result.decision_id}: accepted={result.judgment.accepted}")
    print(f"  {result.judgment.reason}")

    # 4. Promotion is separate from judgment, and rollback is a pointer move.
    if result.judgment.accepted:
        promote(workspace, "task", candidate)
        print(f"promoted: task -> {bundle_hash(candidate)}")
    promote(workspace, "task", initial)
    print(f"rollback: task -> {bundle_hash(resolve(workspace, 'task'))}")

    print("\ndecision log:")
    for decision in read_decisions(workspace):
        print(
            f"  {decision.id} [{decision.level}/{decision.kind}] "
            f"{decision.decision}: {decision.reason}"
        )


if __name__ == "__main__":
    main()
