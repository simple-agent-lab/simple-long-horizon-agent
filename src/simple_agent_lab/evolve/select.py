"""Selection pressure: parent sampling strategies and acceptance policies.

Both seams are deliberately small functions over the archive; anything more
elaborate (novelty-aware sampling, beam search) is a custom
`SelectFn`/`AcceptFn` in the experiment script, not a harness feature.

Parent samplers return a non-empty sequence: `[parent, *inspirations]`. The
parent is what the proposer mutates; inspirations are extra high-fitness
context (ShinkaEvolve's "top-k inspirations") it may borrow ideas from.

`select_islands` shows how far the seam stretches without new record
fields: island membership is *derived from lineage* (seed order, then
inherited from the parent), so the population structure lives entirely in
the selector, stays deterministic, and survives `Archive.load` resumes.
"""

from __future__ import annotations

import random
from typing import Sequence

from .archive import Archive
from .types import Candidate, Decision, Evaluation, EvolutionRecord, SelectFn


def _with_inspirations(
    parent: EvolutionRecord, archive: Archive, inspirations: int
) -> list[EvolutionRecord]:
    others = [r for r in archive.top(inspirations + 1) if r is not parent]
    return [parent, *others[:inspirations]]


def select_best(*, inspirations: int = 0) -> SelectFn:
    """Greedy hill-climbing: always mutate the current best."""

    def select(archive: Archive, rng: random.Random) -> Sequence[EvolutionRecord]:
        best = archive.best()
        if best is None:
            raise ValueError("select_best: archive population is empty")
        return _with_inspirations(best, archive, inspirations)

    return select


def select_uniform(*, inspirations: int = 0) -> SelectFn:
    """Pure exploration: every population member is equally likely."""

    def select(archive: Archive, rng: random.Random) -> Sequence[EvolutionRecord]:
        population = archive.population()
        if not population:
            raise ValueError("select_uniform: archive population is empty")
        return _with_inspirations(rng.choice(population), archive, inspirations)

    return select


def select_weighted(*, power: float = 2.0, inspirations: int = 2) -> SelectFn:
    """Rank-power-law sampling: better candidates are likelier parents.

    Population members are ranked by fitness (worst first) and drawn with
    weight `rank ** power`. `power=0` degenerates to uniform; larger values
    approach greedy — the same exploitation knob as ShinkaEvolve's power-law
    parent sampler, without the offspring-count correction.
    """

    def select(archive: Archive, rng: random.Random) -> Sequence[EvolutionRecord]:
        population = archive.population()
        if not population:
            raise ValueError("select_weighted: archive population is empty")
        ranked = sorted(population, key=lambda r: r.evaluation.fitness)
        weights = [float(rank + 1) ** power for rank in range(len(ranked))]
        parent = rng.choices(ranked, weights=weights, k=1)[0]
        return _with_inspirations(parent, archive, inspirations)

    return select


def accept_correct(
    candidate: Candidate, evaluation: Evaluation, archive: Archive
) -> Decision:
    """Default policy: every correct candidate joins the population.

    Selection pressure then comes entirely from the parent sampler. This is
    the population-based shape (ShinkaEvolve, DGM archives): keeping mediocre
    candidates preserves stepping stones.
    """

    if not evaluation.correct:
        return Decision(False, f"incorrect: {evaluation.error or 'evaluator said so'}")
    return Decision(True, f"correct, fitness {evaluation.fitness:.4f}")


def accept_improves_best(
    candidate: Candidate, evaluation: Evaluation, archive: Archive
) -> Decision:
    """Hill-climb policy: accept only strict improvements over the best so far."""

    if not evaluation.correct:
        return Decision(False, f"incorrect: {evaluation.error or 'evaluator said so'}")
    best = archive.best()
    if best is None:
        return Decision(
            True, f"first correct candidate, fitness {evaluation.fitness:.4f}"
        )
    if evaluation.fitness > best.evaluation.fitness:
        return Decision(
            True,
            f"fitness {evaluation.fitness:.4f} > best {best.evaluation.fitness:.4f}"
            f" ({best.candidate.id})",
        )
    return Decision(
        False,
        f"fitness {evaluation.fitness:.4f} <= best {best.evaluation.fitness:.4f}"
        f" ({best.candidate.id})",
    )


def _island_of(archive: Archive, num_islands: int) -> dict[str, int]:
    """Derive every record's island from lineage (no stored island field).

    Seeds (no parents) are dealt round-robin across islands in arrival
    order; every child lives on its first parent's island. Records are
    causally ordered (parents precede children), so one forward pass
    resolves the whole map — including after a resume.
    """

    islands: dict[str, int] = {}
    seed_count = 0
    for record in archive.records:
        if not record.candidate.parent_ids:
            islands[record.candidate.id] = seed_count % num_islands
            seed_count += 1
        else:
            islands[record.candidate.id] = islands.get(
                record.candidate.parent_ids[0], 0
            )
    return islands


def select_islands(
    *,
    num_islands: int = 4,
    power: float = 2.0,
    inspirations: int = 2,
    migration_interval: int = 10,
) -> SelectFn:
    """Island-model selection: isolated subpopulations with periodic mixing.

    Proposals rotate round-robin across islands (the next record index picks
    the island), the parent is rank-power-law sampled *within* that island,
    and inspirations come from the same island — except every
    `migration_interval`-th proposal, when the globally best records are
    offered instead (migration by inspiration, ShinkaEvolve-style). Islands
    whose population is still empty borrow the whole population, so small
    early runs behave like `select_weighted`.
    """

    if num_islands < 1:
        raise ValueError(f"num_islands must be >= 1, got {num_islands}")

    def select(archive: Archive, rng: random.Random) -> Sequence[EvolutionRecord]:
        population = archive.population()
        if not population:
            raise ValueError("select_islands: archive population is empty")
        islands = _island_of(archive, num_islands)
        island = len(archive.records) % num_islands
        members = [r for r in population if islands[r.candidate.id] == island]
        if not members:
            members = population

        ranked = sorted(members, key=lambda r: r.evaluation.fitness)
        weights = [float(rank + 1) ** power for rank in range(len(ranked))]
        parent = rng.choices(ranked, weights=weights, k=1)[0]

        migrate = (
            migration_interval > 0
            and len(archive.records) % migration_interval == migration_interval - 1
        )
        pool = population if migrate else members
        top_others = sorted(
            (r for r in pool if r is not parent),
            key=lambda r: r.evaluation.fitness,
            reverse=True,
        )[:inspirations]
        return [parent, *top_others]

    return select
