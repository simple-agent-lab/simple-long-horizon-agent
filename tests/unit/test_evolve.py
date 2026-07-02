"""Evolution harness unit tests: archive, selection, loop (deterministic)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from simple_agent_lab.evolve import (
    Archive,
    Candidate,
    Evaluation,
    EvolutionBudgets,
    EvolutionRecord,
    Proposal,
    accept_correct,
    accept_improves_best,
    record_from_dict,
    record_to_dict,
    run_evolution,
    select_best,
    select_uniform,
    select_weighted,
)


def make_record(
    candidate_id: str,
    fitness: float,
    *,
    accepted: bool = True,
    correct: bool = True,
) -> EvolutionRecord:
    return EvolutionRecord(
        candidate=Candidate(id=candidate_id, payload={"text": candidate_id}),
        evaluation=Evaluation(fitness=fitness, correct=correct),
        accepted=accepted,
        reason="test",
    )


def count_up_propose(
    parents: Sequence[EvolutionRecord], rng: random.Random
) -> Proposal:
    """Deterministic improvement: child value = parent value + 1."""

    return Proposal(payload={"value": parents[0].candidate.payload["value"] + 1})


def value_evaluate(candidate: Candidate) -> Evaluation:
    return Evaluation(fitness=float(candidate.payload["value"]))


class ArchiveTest(unittest.TestCase):
    def test_population_excludes_rejected_and_incorrect(self):
        archive = Archive()
        archive.add(make_record("c0000", 0.5))
        archive.add(make_record("c0001", 0.9, accepted=False))
        archive.add(make_record("c0002", 0.8, correct=False))
        self.assertEqual([r.candidate.id for r in archive.population()], ["c0000"])

    def test_best_and_top_order_by_fitness(self):
        archive = Archive()
        for candidate_id, fitness in [("c0000", 0.2), ("c0001", 0.9), ("c0002", 0.5)]:
            archive.add(make_record(candidate_id, fitness))
        self.assertEqual(archive.best().candidate.id, "c0001")
        self.assertEqual([r.candidate.id for r in archive.top(2)], ["c0001", "c0002"])

    def test_record_round_trips_through_dict(self):
        record = EvolutionRecord(
            candidate=Candidate(
                id="c0003",
                payload={"prompt": "hi", "config": {"k": 1}},
                parent_ids=("c0001", "c0002"),
                operator="llm_mutate",
                generation=4,
                note="tightened wording",
            ),
            evaluation=Evaluation(
                fitness=0.75,
                correct=True,
                metrics={"solved": 3},
                feedback="task 2 failed",
                error="",
            ),
            accepted=True,
            reason="correct, fitness 0.7500",
        )
        self.assertEqual(record_from_dict(record_to_dict(record)), record)

    def test_jsonl_persist_and_load_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "archive.jsonl"
            archive = Archive(path=path)
            archive.add(make_record("c0000", 0.1))
            archive.add(make_record("c0001", 0.7))

            loaded = Archive.load(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.next_candidate_id(), "c0002")
            self.assertEqual(loaded.best().candidate.id, "c0001")

            # Appends after load land in the same file.
            loaded.add(make_record("c0002", 0.9))
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[-1])["candidate"]["id"], "c0002")


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.archive = Archive()
        for candidate_id, fitness in [
            ("c0000", 0.1),
            ("c0001", 0.9),
            ("c0002", 0.5),
        ]:
            self.archive.add(make_record(candidate_id, fitness))

    def test_select_best_returns_best_plus_inspirations(self):
        parents = select_best(inspirations=1)(self.archive, random.Random(0))
        self.assertEqual([r.candidate.id for r in parents], ["c0001", "c0002"])

    def test_select_weighted_prefers_high_fitness(self):
        select = select_weighted(power=3.0, inspirations=0)
        rng = random.Random(0)
        picks = [select(self.archive, rng)[0].candidate.id for _ in range(200)]
        self.assertGreater(picks.count("c0001"), picks.count("c0000"))

    def test_select_uniform_never_returns_parent_as_inspiration(self):
        select = select_uniform(inspirations=2)
        for trial in range(20):
            parents = select(self.archive, random.Random(trial))
            self.assertEqual(len(parents), len({r.candidate.id for r in parents}))

    def test_selectors_raise_on_empty_population(self):
        empty = Archive()
        for select in (select_best(), select_uniform(), select_weighted()):
            with self.assertRaises(ValueError):
                select(empty, random.Random(0))


class AcceptanceTest(unittest.TestCase):
    def test_accept_correct_rejects_incorrect_with_reason(self):
        archive = Archive()
        decision = accept_correct(
            Candidate(id="c0000", payload={}),
            Evaluation(fitness=0.0, correct=False, error="boom"),
            archive,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("boom", decision.reason)

    def test_accept_improves_best_compares_against_archive(self):
        archive = Archive()
        archive.add(make_record("c0000", 0.5))
        candidate = Candidate(id="c0001", payload={})
        better = accept_improves_best(candidate, Evaluation(fitness=0.6), archive)
        worse = accept_improves_best(candidate, Evaluation(fitness=0.4), archive)
        self.assertTrue(better.accepted)
        self.assertFalse(worse.accepted)
        self.assertIn("c0000", worse.reason)


class RunEvolutionTest(unittest.TestCase):
    def test_reaches_target_and_stops(self):
        result = run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=value_evaluate,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=100, target_fitness=5.0),
        )
        self.assertEqual(result.status, "target_reached")
        self.assertEqual(result.best.evaluation.fitness, 5.0)
        self.assertEqual(len(result.archive), 6)  # seed + 5 improvements

    def test_budget_bounds_total_archive_size(self):
        result = run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=value_evaluate,
            budgets=EvolutionBudgets(max_candidates=4),
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(len(result.archive), 4)
        self.assertEqual(result.candidates_added, 4)

    def test_lineage_and_generation_recorded(self):
        result = run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=value_evaluate,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=3),
        )
        child = result.archive.records[-1]
        parent = result.archive.records[-2]
        self.assertEqual(child.candidate.parent_ids, (parent.candidate.id,))
        self.assertEqual(child.candidate.generation, parent.candidate.generation + 1)

    def test_evaluator_exception_becomes_rejected_record(self):
        def explode(candidate: Candidate) -> Evaluation:
            if candidate.payload["value"] == 1:
                raise RuntimeError("cannot score")
            return value_evaluate(candidate)

        result = run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=explode,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=3),
        )
        failed = result.archive.records[1]
        self.assertFalse(failed.accepted)
        self.assertFalse(failed.evaluation.correct)
        self.assertIn("cannot score", failed.evaluation.error)
        # The loop kept going after the failure.
        self.assertEqual(len(result.archive), 3)

    def test_proposer_exception_becomes_propose_error_record(self):
        calls = {"n": 0}

        def flaky_propose(parents, rng):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("no editable fields")
            return count_up_propose(parents, rng)

        result = run_evolution(
            seeds=[{"value": 0}],
            propose=flaky_propose,
            evaluate=value_evaluate,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=3),
        )
        failed = result.archive.records[1]
        self.assertEqual(failed.candidate.operator, "propose_error")
        self.assertIn("no editable fields", failed.evaluation.error)
        self.assertEqual(result.archive.best().evaluation.fitness, 1.0)

    def test_resume_skips_seeding_and_continues_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            first = run_evolution(
                seeds=[{"value": 0}],
                propose=count_up_propose,
                evaluate=value_evaluate,
                select=select_best(),
                budgets=EvolutionBudgets(max_candidates=3),
                archive=Archive(path=path),
            )
            self.assertEqual(first.status, "budget_exhausted")

            resumed = run_evolution(
                seeds=[{"value": 0}],  # ignored: archive already has records
                propose=count_up_propose,
                evaluate=value_evaluate,
                select=select_best(),
                budgets=EvolutionBudgets(max_candidates=5),
                archive=Archive.load(path),
            )
            self.assertEqual(resumed.candidates_added, 2)
            ids = [r.candidate.id for r in resumed.archive.records]
            self.assertEqual(ids, ["c0000", "c0001", "c0002", "c0003", "c0004"])
            # Fitness kept climbing from the reloaded best, not from scratch.
            self.assertEqual(resumed.best.evaluation.fitness, 4.0)

    def test_all_seeds_rejected_raises(self):
        with self.assertRaises(ValueError):
            run_evolution(
                seeds=[{"value": 0}],
                propose=count_up_propose,
                evaluate=lambda c: Evaluation(fitness=0.0, correct=False, error="bad"),
                budgets=EvolutionBudgets(max_candidates=10),
            )

    def test_no_seeds_and_empty_archive_raises(self):
        with self.assertRaises(ValueError):
            run_evolution(
                seeds=[],
                propose=count_up_propose,
                evaluate=value_evaluate,
            )

    def test_abort_stops_promptly(self):
        countdown = {"n": 3}

        def abort() -> bool:
            countdown["n"] -= 1
            return countdown["n"] <= 0

        result = run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=value_evaluate,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=100),
            abort=abort,
        )
        self.assertEqual(result.status, "aborted")
        self.assertLess(len(result.archive), 100)

    def test_on_record_sees_every_record(self):
        seen: list[str] = []
        run_evolution(
            seeds=[{"value": 0}],
            propose=count_up_propose,
            evaluate=value_evaluate,
            select=select_best(),
            budgets=EvolutionBudgets(max_candidates=3),
            on_record=lambda r: seen.append(r.candidate.id),
        )
        self.assertEqual(seen, ["c0000", "c0001", "c0002"])


if __name__ == "__main__":
    unittest.main()
