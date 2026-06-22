from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from recipes.dgm.algorithm import open_ended
from simple_agent_lab.evolution.kernel import log, store
from simple_agent_lab.evolution.types import Proposal, Slice, Verdict


class _FakeRun:
    def __init__(self, instance_id: str, reward: float, run_id: str) -> None:
        self.instance_id = instance_id
        self.reward = reward
        self.run_id = run_id


class RunRoundTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()
        seed = store.stage(
            self.ws, base=None, edits={"scaffold/agent_scaffold.py": "x = 0\n"}
        )
        store.promote(self.ws, seed)
        self.seed_hash = seed.hash
        self.slice_ = Slice("train", ({"instance_id": "i1"}, {"instance_id": "i2"}))

        self._proposal_counter = 0
        self._counter_lock = threading.Lock()
        self.rollout_peak = 0
        self._rollout_active = 0
        self.criterion_peak = 0
        self._criterion_active = 0

    def _components(self):
        def rollout(version, slice_):
            with self._counter_lock:
                self._rollout_active += 1
                self.rollout_peak = max(self.rollout_peak, self._rollout_active)
            time.sleep(0.02)
            with self._counter_lock:
                self._rollout_active -= 1
            reward = 0.5 if version.hash == self.seed_hash else 0.75
            return [
                _FakeRun(str(inst["instance_id"]), reward, run_id=version.hash)
                for inst in slice_.instances
            ]

        def strategy(ctx):
            with self._counter_lock:
                self._proposal_counter += 1
                n = self._proposal_counter
            return Proposal(
                base="",
                edits={"scaffold/agent_scaffold.py": f"x = {n}\n"},
                note=f"branch-{n}",
                kind="sal-meta-scaffold",
            )

        def criterion(base_scores, cand_scores):
            with self._counter_lock:
                self._criterion_active += 1
                self.criterion_peak = max(self.criterion_peak, self._criterion_active)
            time.sleep(0.01)
            with self._counter_lock:
                self._criterion_active -= 1
            b = sum(d["reward"] for d in base_scores.values()) / len(base_scores)
            c = sum(d["reward"] for d in cand_scores.values()) / len(cand_scores)
            return Verdict(c >= b, f"{b}->{c}", {"reward": c - b})

        return SimpleNamespace(
            rollout=rollout,
            reward=lambda run: run.reward,
            strategy=strategy,
            criterion=criterion,
        )

    def test_run_round_evaluates_all_branches(self) -> None:
        decisions = open_ended.run_round(
            self.ws, self._components(), self.slice_, branches=3
        )
        self.assertEqual(len(decisions), 3)
        logged = log.read(self.ws)
        self.assertEqual(len(logged), 3)

    def test_run_round_skips_failed_proposal_branch(self) -> None:
        components = self._components()
        original_strategy = components.strategy

        def strategy(ctx):
            with self._counter_lock:
                self._proposal_counter += 1
                n = self._proposal_counter
            if n == 1:
                raise ValueError("bad model response")
            return original_strategy(ctx)

        components.strategy = strategy

        decisions = open_ended.run_round(
            self.ws,
            components,
            self.slice_,
            branches=3,
            on_proposal_error=lambda e: None,
        )

        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(log.read(self.ws)), 2)

    def test_rollouts_run_concurrently_but_tail_is_serialized(self) -> None:
        open_ended.run_round(self.ws, self._components(), self.slice_, branches=3)
        self.assertGreater(self.rollout_peak, 1)
        self.assertEqual(self.criterion_peak, 1)

    def test_best_accepted_candidate_is_promoted(self) -> None:
        open_ended.run_round(self.ws, self._components(), self.slice_, branches=3)
        current = store.current(self.ws)
        self.assertNotEqual(current.hash, self.seed_hash)

    def test_run_evolution_runs_rounds_times_branches(self) -> None:
        decisions = open_ended.run_evolution(
            self.ws, self._components(), self.slice_, rounds=2, branches=2
        )
        self.assertEqual(len(decisions), 4)


if __name__ == "__main__":
    unittest.main()
