"""ShinkaEvolve-style code evolution: mutate only inside EVOLVE-BLOCK markers.

The payload is one Python source file whose mutable region is fenced by
`# EVOLVE-BLOCK-START/END`; the scaffold (the scoring interface) is
immutable and `check_immutable_regions` rejects any proposal that touches
it. Each candidate runs in a fresh subprocess with a timeout, so broken or
runaway proposals become rejected records instead of crashing the run.

The task: fit `f(x)` to a hidden quadratic from sampled points.

Run offline (deterministic coefficient-jitter proposer):
    uv run python scripts/run_code_evolution_demo.py

Run with a real model proposing code edits (uses the provider env, see
`.env.example`):
    uv run python scripts/run_code_evolution_demo.py --live
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from typing import Sequence

from simple_agent_lab.evolve import (
    Candidate,
    Evaluation,
    EvolutionBudgets,
    EvolutionRecord,
    Proposal,
    ProposeFn,
    accept_correct,
    ask_from_provider,
    check_immutable_regions,
    llm_propose,
    run_evolution,
    select_weighted,
)

SEED_SOURCE = '''\
# EVOLVE-BLOCK-START
def f(x):
    """Approximate the hidden function. Improve me."""
    return 1.0 * x + 0.0
# EVOLVE-BLOCK-END


def score(samples):
    """Immutable scoring interface: mean squared error over samples."""
    return sum((f(x) - y) ** 2 for x, y in samples) / len(samples)
'''

# The hidden function candidates must approximate: y = 3x^2 + 2x + 1.
SAMPLES = [(x / 2, 3 * (x / 2) ** 2 + 2 * (x / 2) + 1) for x in range(-8, 9)]

EVAL_TIMEOUT_SECONDS = 10.0


def evaluate(candidate: Candidate) -> Evaluation:
    """Score the candidate's `f` in a subprocess; fitness = 1 / (1 + MSE)."""

    source = candidate.payload["code"]
    check_immutable_regions(SEED_SOURCE, source)  # reject scaffold edits
    harness = (
        f"{source}\nimport json\nprint(json.dumps({{'mse': score({SAMPLES!r})}}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=EVAL_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        return Evaluation(
            fitness=0.0, correct=False, error=completed.stderr.strip()[-1000:]
        )
    mse = float(json.loads(completed.stdout)["mse"])
    return Evaluation(
        fitness=1.0 / (1.0 + mse),
        metrics={"mse": mse},
        feedback=f"mean squared error {mse:.4f} over {len(SAMPLES)} samples",
    )


def jitter_propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
    """Offline proposer: nudge the numeric literals in the mutable block.

    Knows nothing about the task beyond "the block contains numbers" — a
    stand-in for the LLM so the demo runs deterministically with no network.
    """

    source = parents[0].candidate.payload["code"]

    def nudge(match: re.Match[str]) -> str:
        return f"{float(match.group(0)) + rng.uniform(-0.5, 0.5):.3f}"

    start = source.index("# EVOLVE-BLOCK-START")
    end = source.index("# EVOLVE-BLOCK-END")
    block = re.sub(r"-?\d+\.\d+", nudge, source[start:end])
    # Occasionally raise the polynomial degree so improvement is reachable.
    if "x * x" not in block and rng.random() < 0.3:
        block = block.replace("return ", "return 1.0 * x * x + ", 1)
    return Proposal(
        payload={"code": source[:start] + block + source[end:]},
        operator="jitter",
        note="nudged numeric coefficients",
    )


def live_propose() -> ProposeFn:
    from simple_agent_lab.llm import provider_from_env

    ask = ask_from_provider(provider_from_env())
    return llm_propose(
        ask,
        task=(
            "Rewrite f(x) so score(samples) — the mean squared error against "
            "a hidden smooth function — gets as small as possible. Edit only "
            "the code between the EVOLVE-BLOCK markers; keep the rest intact."
        ),
        fields=["code"],
        guidance="Return the complete file in the code block, markers included.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="LLM proposer via env")
    parser.add_argument("--budget", type=int, default=200, help="max archive size")
    parser.add_argument("--seed", type=int, default=3, help="rng seed")
    args = parser.parse_args()

    best_seen = {"fitness": -1.0}

    def show_progress(record: EvolutionRecord) -> None:
        if record.accepted and record.evaluation.fitness > best_seen["fitness"]:
            best_seen["fitness"] = record.evaluation.fitness
            mse = record.evaluation.metrics.get("mse", float("nan"))
            print(
                f"{record.candidate.id}  gen {record.candidate.generation:>3}  "
                f"mse {mse:10.4f}  ({record.candidate.operator})"
            )

    result = run_evolution(
        seeds=[{"code": SEED_SOURCE}],
        propose=live_propose() if args.live else jitter_propose,
        evaluate=evaluate,
        select=select_weighted(power=3.0, inspirations=1),
        accept=accept_correct,
        budgets=EvolutionBudgets(max_candidates=args.budget, target_fitness=0.999),
        rng_seed=args.seed,
        on_record=show_progress,
    )

    print(f"\nstatus: {result.status} after {len(result.archive)} candidates")
    if result.best is not None:
        print(f"best mse: {result.best.evaluation.metrics.get('mse'):.6f}")
        print("\nbest candidate source:\n")
        print(result.best.candidate.payload["code"])


if __name__ == "__main__":
    main()
