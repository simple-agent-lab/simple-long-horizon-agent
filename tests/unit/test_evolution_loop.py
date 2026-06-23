from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from simple_agent_lab.evolution.components.criterion import (
    guarded,
    improve,
    not_worse,
    promote_not_worse,
)
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.kernel import loop, store
from simple_agent_lab.evolution.types import Context, Proposal, Run, Slice, Version


@dataclass(frozen=True)
class Components:
    rollout: Callable[[Version, Slice], Sequence[Run]]
    reward: Callable[[Run], float]
    strategy: Callable[[Context], "Proposal | None"]
    criterion: Callable


def stub_rollout(rewards_by_prompt: dict[str, float], runs_root: Path):
    def rollout(version: Version, slice_: Slice) -> list[Run]:
        reward = rewards_by_prompt[version.read("prompt.md")]
        run_dir = runs_root / f"{version.hash}-{slice_.sha}" / "i1"
        (run_dir / "out").mkdir(parents=True, exist_ok=True)
        (run_dir / "out" / "result.json").write_text(json.dumps({"reward": reward}))
        return [Run(run_dir)]

    return rollout


def multi_stub_rollout(metrics_by_prompt: dict[str, dict[str, float]], runs_root: Path):
    """Like ``stub_rollout`` but writes several dims into ``result.json``."""

    def rollout(version: Version, slice_: Slice) -> list[Run]:
        metrics = metrics_by_prompt[version.read("prompt.md")]
        run_dir = runs_root / f"{version.hash}-{slice_.sha}" / "i1"
        (run_dir / "out").mkdir(parents=True, exist_ok=True)
        (run_dir / "out" / "result.json").write_text(json.dumps(metrics))
        return [Run(run_dir)]

    return rollout


def multi_reward(run: Run) -> dict[str, float]:
    """A dict-returning reward exercising loop.score's Mapping branch."""

    return {
        "reward": run.reward if run.reward is not None else 0.0,
        "cost": float(run.result.get("cost", 0.0)),
    }


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)
        initial = store.stage(self.ws, base=None, edits={"prompt.md": "weak"})
        store.promote(self.ws, initial)
        self.slice = Slice("demo", ({"instance_id": "i1"},))

    def _components(self, strategy) -> Components:
        return Components(
            rollout=stub_rollout({"weak": 0.3, "strong": 0.7}, self.ws / "runs"),
            reward=result_key,
            strategy=strategy,
            criterion=improve("reward"),
        )

    def test_accepted_proposal_promotes(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(
                edits={"prompt.md": "strong"}, note="try strong", kind="prompt"
            )

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertTrue(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "strong")
        self.assertAlmostEqual(decision.deltas["reward"], 0.4)

    def test_rejected_proposal_keeps_current(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(
                edits={"prompt.md": "weak"}, note="no change", kind="prompt"
            )

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertFalse(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "weak")

    def test_unchanged_candidate_is_logged_as_no_op_without_promotion(self) -> None:
        rollout_calls: list[str] = []

        def rollout(version: Version, slice_: Slice) -> list[Run]:
            rollout_calls.append(version.hash)
            run_dir = self.ws / "runs" / f"{version.hash}-{slice_.sha}" / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(
                json.dumps({"reward": 1.0}), encoding="utf-8"
            )
            return [Run(run_dir)]

        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={}, note="empty edit", kind="prompt")

        components = Components(
            rollout=rollout,
            reward=result_key,
            strategy=strategy,
            criterion=promote_not_worse("reward"),
        )

        decision = loop.step(self.ws, components, self.slice)

        self.assertFalse(decision.accepted)
        self.assertIn("no-op", decision.reason)
        self.assertEqual(decision.baseline["hash"], decision.candidate["hash"])
        self.assertEqual(store.current(self.ws).hash, decision.baseline["hash"])
        self.assertEqual(rollout_calls, [decision.baseline["hash"]])

    def test_no_proposal_returns_none(self) -> None:
        decision = loop.step(self.ws, self._components(lambda ctx: None), self.slice)
        self.assertIsNone(decision)

    def test_unknown_base_raises(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, base="deadbeef")

        with self.assertRaises(ValueError):
            loop.step(self.ws, self._components(strategy), self.slice)

    def test_path_like_proposal_base_is_rejected(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, base="../pointers")

        with self.assertRaises(ValueError):
            loop.step(self.ws, self._components(strategy), self.slice)

    def test_proposal_base_is_comparison_baseline(self) -> None:
        archived_parent = store.stage(
            self.ws, base=store.current(self.ws), edits={"prompt.md": "medium"}
        )

        def strategy(ctx: Context) -> Proposal:
            return Proposal(
                edits={"prompt.md": "strong"},
                base=archived_parent.hash,
                note="branch from archived parent",
                kind="prompt",
            )

        components = Components(
            rollout=stub_rollout(
                {"weak": 0.3, "medium": 0.5, "strong": 0.7}, self.ws / "runs"
            ),
            reward=result_key,
            strategy=strategy,
            criterion=improve("reward"),
        )

        decision = loop.step(self.ws, components, self.slice)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.baseline["hash"], archived_parent.hash)
        self.assertAlmostEqual(decision.baseline["scores"]["reward"], 0.5)
        self.assertAlmostEqual(decision.deltas["reward"], 0.2)

    def test_empty_rollout_raises(self) -> None:
        components = Components(
            rollout=lambda version, slice_: [],
            reward=result_key,
            strategy=lambda ctx: Proposal(edits={"prompt.md": "strong"}, kind="prompt"),
            criterion=improve("reward"),
        )
        with self.assertRaises(ValueError):
            loop.step(self.ws, components, self.slice)

    def _multi_components(
        self, strategy, metrics: dict[str, dict[str, float]], *, tol: float
    ) -> Components:
        return Components(
            rollout=multi_stub_rollout(metrics, self.ws / "runs"),
            reward=multi_reward,
            strategy=strategy,
            criterion=guarded(improve("reward"), [not_worse("cost", tol=tol)]),
        )

    def test_multi_dim_reward_guard_passes(self) -> None:
        # dict reward -> loop.score(Mapping branch) -> multi-dim guarded criterion.
        # reward climbs (0.3 -> 0.7); cost holds steady so the guard holds -> accept.
        def strategy(ctx: Context) -> Proposal:
            return Proposal(
                edits={"prompt.md": "strong"}, note="try strong", kind="prompt"
            )

        metrics = {
            "weak": {"reward": 0.3, "cost": 0.5},
            "strong": {"reward": 0.7, "cost": 0.5},
        }
        decision = loop.step(
            self.ws, self._multi_components(strategy, metrics, tol=0.0), self.slice
        )
        self.assertTrue(decision.accepted)
        self.assertIn("reward", decision.deltas)
        self.assertIn("cost", decision.deltas)
        self.assertAlmostEqual(decision.deltas["reward"], 0.4)
        self.assertAlmostEqual(decision.deltas["cost"], 0.0)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "strong")

    def test_multi_dim_reward_guard_fails(self) -> None:
        # reward still climbs, but the cost dim trips not_worse -> guard rejects.
        def strategy(ctx: Context) -> Proposal:
            return Proposal(
                edits={"prompt.md": "strong"}, note="try strong", kind="prompt"
            )

        metrics = {
            "weak": {"reward": 0.3, "cost": 0.5},
            "strong": {"reward": 0.7, "cost": 0.1},
        }
        decision = loop.step(
            self.ws, self._multi_components(strategy, metrics, tol=0.0), self.slice
        )
        self.assertFalse(decision.accepted)
        self.assertIn("reward", decision.deltas)
        self.assertIn("cost", decision.deltas)
        self.assertAlmostEqual(decision.deltas["cost"], -0.4)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "weak")

    def test_run_collects_only_non_declined_steps(self) -> None:
        # Improve once (accepted), then decline -> run(n=2) yields a single Decision.
        calls = {"n": 0}

        def strategy(ctx: Context) -> "Proposal | None":
            calls["n"] += 1
            if calls["n"] == 1:
                return Proposal(edits={"prompt.md": "strong"}, kind="prompt")
            return None

        result = loop.run(self.ws, self._components(strategy), self.slice, n=2)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].accepted)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
