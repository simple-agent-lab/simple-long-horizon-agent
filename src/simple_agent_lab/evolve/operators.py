"""Mutation-operator combinators: mix operators, crossover, targeted rewrite.

ShinkaEvolve samples a patch type per proposal (diff / full rewrite /
crossover, each with a probability); this module is that idea expressed over
the existing `ProposeFn` seam. An "operator" is just a `ProposeFn` whose
`Proposal.operator` labels it — the archive then shows which operator
produced every candidate, so operator effectiveness is measurable from the
run's own records.
"""

from __future__ import annotations

import random
from typing import Sequence

from .propose import (
    AskFn,
    build_mutation_prompt,
    parse_fields,
    proposal_note,
    render_fields,
)
from .types import EvolutionRecord, Proposal, ProposeFn


def mix_operators(operators: Sequence[tuple[ProposeFn, float]]) -> ProposeFn:
    """Sample one operator per proposal, weighted.

    `operators` is `[(propose_fn, weight), ...]`. The chosen operator's own
    `Proposal.operator` label is kept, so the archive attributes each child
    to the operator that made it.
    """

    if not operators:
        raise ValueError("mix_operators needs at least one operator")
    fns = [fn for fn, _ in operators]
    weights = [weight for _, weight in operators]
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(
            f"operator weights must be >= 0 with a positive sum: {weights}"
        )

    def propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
        chosen = rng.choices(fns, weights=weights, k=1)[0]
        return chosen(parents, rng)

    return propose


def crossover_propose(
    ask: AskFn,
    *,
    task: str,
    fields: Sequence[str],
    guidance: str = "",
    operator: str = "llm_crossover",
) -> ProposeFn:
    """Combine the parent with its strongest inspiration into one child.

    Needs at least two parents (parent + one inspiration, i.e. a `SelectFn`
    with `inspirations >= 1`). With only one, it degrades to a plain
    mutation prompt — a young population should not make crossover a hard
    error — and labels the child `<operator>_solo`.
    """

    def propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
        if len(parents) < 2:
            prompt = build_mutation_prompt(
                parents, task=task, fields=fields, guidance=guidance
            )
            response = ask(prompt)
            changed = parse_fields(response, fields)
            payload = {**dict(parents[0].candidate.payload), **changed}
            return Proposal(
                payload=payload,
                operator=f"{operator}_solo",
                note=proposal_note(response),
            )

        first, second = parents[0], parents[1]
        lines = [
            "You are combining two candidates from an evolutionary search "
            "into one child that keeps the strengths of both.",
            "",
            f"Goal: {task}",
        ]
        if guidance:
            lines += ["", f"Guidance: {guidance}"]
        lines += [
            "",
            f"Candidate A (fitness {first.evaluation.fitness:.4f}):",
            render_fields(dict(first.candidate.payload), fields),
        ]
        if first.evaluation.feedback:
            lines += ["", "Feedback on A:", first.evaluation.feedback]
        lines += [
            "",
            f"Candidate B (fitness {second.evaluation.fitness:.4f}):",
            render_fields(dict(second.candidate.payload), fields),
        ]
        if second.evaluation.feedback:
            lines += ["", "Feedback on B:", second.evaluation.feedback]
        lines += [
            "",
            "Produce ONE combined child. First explain what you took from "
            "each, then return every field you changed exactly in this "
            "format (unchanged fields inherit from candidate A):",
            "",
            "### <field name>",
            "```",
            "<new content>",
            "```",
        ]
        response = ask("\n".join(lines))
        changed = parse_fields(response, fields)
        payload = {**dict(first.candidate.payload), **changed}
        return Proposal(
            payload=payload, operator=operator, note=proposal_note(response)
        )

    return propose
