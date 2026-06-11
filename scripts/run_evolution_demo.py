"""The evolution framework from a researcher's seat, deterministically.

Everything a user touches is in this file: a rollout (here a stub standing
in for a real eval suite), an optional reward function, and a strategy
function. Versioning, comparison, the experiment log, and rollback are the
framework's job. Design: docs/design/20260610-evolution-framework-spec.md.

Usage:
    PYTHONPATH=src python scripts/run_evolution_demo.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from simple_agent_lab.evolution import Bundle, EpisodeContext, EvalSlice, Lab, Proposal, Run


def stub_rollout(runs_root: Path):
    """Stand-in for a real eval suite: reward 0.4, +0.3 when the playbook
    mentions checking pytest fixtures. Swap for evolution.rollout.dataset_rollout
    to run real containerized suites."""

    def rollout(bundle: Bundle, slice_: EvalSlice, run_id: str) -> list[Run]:
        reward = 0.4 + (0.3 if "pytest fixtures" in bundle.read("playbook.md") else 0.0)
        instance = runs_root / run_id / "demo-1"
        (instance / "out").mkdir(parents=True, exist_ok=True)
        (instance / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [Run(instance)]

    return rollout


def my_strategy(ctx: EpisodeContext) -> Proposal | None:
    """A 10-line strategy: when runs underperform, add one playbook bullet."""

    weak = [r for r in ctx.runs if r.reward is not None and r.reward < 0.5]
    if not weak:
        return None
    return ctx.propose(
        kind="playbook",
        edits={
            "playbook.md": ctx.current.read("playbook.md")
            + "- When tests fail on setup, check pytest fixtures first.\n"
        },
        note="playbook bullet: check pytest fixtures before editing tests",
        evidence=[r.ref for r in weak],  # workspace-relative refs, never paths
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="evolution-demo-"))
    print(f"workspace: {workspace}\n")

    lab = Lab(
        workspace,
        rollout=stub_rollout(workspace / "runs"),
        seed={"prompt.md": "You are a careful software agent."},
    )

    print(lab.step(my_strategy).text)  # observe -> propose -> compare -> promote
    print(lab.step(my_strategy).text)  # nothing left to fix: no proposal

    print("\nexperiment log:")
    print(lab.history())

    print(f"\n{lab.rollback()}")  # pointers move; nothing is deleted


if __name__ == "__main__":
    main()
