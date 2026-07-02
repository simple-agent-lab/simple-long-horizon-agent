"""Smallest possible evolution run: no model, no network, one screen of code.

Evolves a string toward a hidden target with random point mutations, purely
to show the harness mechanics: seeds -> propose -> evaluate -> accept ->
archive, plus resume. Replace `propose` with `llm_propose(...)` and
`evaluate` with a real task to turn this scaffold into an experiment.

Run:
    uv run python scripts/run_evolve_demo.py
    uv run python scripts/run_evolve_demo.py --archive /tmp/demo-archive.jsonl
"""

from __future__ import annotations

import argparse
import random
import string
from typing import Sequence

from simple_agent_lab.evolve import (
    Archive,
    Candidate,
    Evaluation,
    EvolutionBudgets,
    EvolutionRecord,
    Proposal,
    accept_improves_best,
    run_evolution,
    select_best,
)

TARGET = "evolving agents, simply"
ALPHABET = string.ascii_lowercase + " ,"


def propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
    """Mutate one random character of the parent's text."""

    text = list(parents[0].candidate.payload["text"])
    position = rng.randrange(len(text))
    text[position] = rng.choice(ALPHABET)
    return Proposal(
        payload={"text": "".join(text)},
        operator="point_mutation",
        note=f"changed position {position}",
    )


def evaluate(candidate: Candidate) -> Evaluation:
    """Fitness = fraction of characters matching the hidden target."""

    text = candidate.payload["text"]
    matches = sum(a == b for a, b in zip(text, TARGET))
    fitness = matches / len(TARGET)
    return Evaluation(
        fitness=fitness, feedback=f"{matches}/{len(TARGET)} characters match"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4000, help="max archive size")
    parser.add_argument("--seed", type=int, default=7, help="rng seed")
    parser.add_argument("--archive", default="", help="persist/resume archive JSONL")
    args = parser.parse_args()

    if args.archive:
        try:
            archive = Archive.load(args.archive)
            print(f"resuming archive with {len(archive)} records")
        except FileNotFoundError:
            archive = Archive(path=args.archive)
    else:
        archive = Archive()

    def show_improvements(record: EvolutionRecord) -> None:
        if record.accepted:
            print(
                f"{record.candidate.id}  gen {record.candidate.generation:>4}  "
                f"fitness {record.evaluation.fitness:.3f}  "
                f"{record.candidate.payload['text']!r}"
            )

    result = run_evolution(
        seeds=[{"text": "x" * len(TARGET)}],
        propose=propose,
        evaluate=evaluate,
        select=select_best(),  # + accept_improves_best = plain hill climbing
        accept=accept_improves_best,
        budgets=EvolutionBudgets(max_candidates=args.budget, target_fitness=1.0),
        rng_seed=args.seed,
        archive=archive,
        on_record=show_improvements,
    )

    best = result.best
    print(f"\nstatus: {result.status} after {len(result.archive)} candidates")
    if best is not None:
        print(f"best:   {best.candidate.id} {best.candidate.payload['text']!r}")
        print(f"        fitness {best.evaluation.fitness:.3f} — {best.reason}")


if __name__ == "__main__":
    main()
