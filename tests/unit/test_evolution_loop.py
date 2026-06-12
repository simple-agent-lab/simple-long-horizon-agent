from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from simple_agent_lab.evolution.components.criterion import improve
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
            return Proposal(edits={"prompt.md": "strong"}, note="try strong", kind="prompt")

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertTrue(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "strong")
        self.assertAlmostEqual(decision.deltas["reward"], 0.4)

    def test_rejected_proposal_keeps_current(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "weak"}, note="no change", kind="prompt")

        decision = loop.step(self.ws, self._components(strategy), self.slice)
        self.assertFalse(decision.accepted)
        self.assertEqual(store.current(self.ws).read("prompt.md"), "weak")

    def test_no_proposal_returns_none(self) -> None:
        decision = loop.step(self.ws, self._components(lambda ctx: None), self.slice)
        self.assertIsNone(decision)

    def test_unknown_base_raises(self) -> None:
        def strategy(ctx: Context) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, base="deadbeef")

        with self.assertRaises(ValueError):
            loop.step(self.ws, self._components(strategy), self.slice)

    def test_empty_rollout_raises(self) -> None:
        components = Components(
            rollout=lambda version, slice_: [],
            reward=result_key,
            strategy=lambda ctx: Proposal(edits={"prompt.md": "strong"}, kind="prompt"),
            criterion=improve("reward"),
        )
        with self.assertRaises(ValueError):
            loop.step(self.ws, components, self.slice)


if __name__ == "__main__":
    unittest.main()
