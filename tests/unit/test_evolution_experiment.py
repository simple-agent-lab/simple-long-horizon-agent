# tests/unit/test_evolution_experiment.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution import Experiment, Proposal
from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key


class ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()

    def _experiment(self) -> Experiment:
        def rollout(version, slice_):
            from simple_agent_lab.evolution.types import Run

            reward = 0.7 if version.read("prompt.md") == "strong" else 0.3
            run_dir = self.ws / "runs" / f"{version.hash}-{slice_.sha}" / "i1"
            (run_dir / "out").mkdir(parents=True, exist_ok=True)
            (run_dir / "out" / "result.json").write_text(json.dumps({"reward": reward}))
            return [Run(run_dir)]

        return Experiment(
            self.ws,
            rollout=rollout,
            reward=result_key,
            criterion=improve("reward"),
            slice_id="demo",
            instances=({"instance_id": "i1"},),
            seed={"prompt.md": "weak"},
        )

    def test_end_to_end_promote_history_rollback(self) -> None:
        exp = self._experiment()

        def to_strong(ctx) -> Proposal:
            return Proposal(edits={"prompt.md": "strong"}, note="upgrade", kind="prompt")

        decision = exp.step(to_strong)
        self.assertTrue(decision.accepted)
        self.assertEqual(exp.current().read("prompt.md"), "strong")

        def noop(ctx) -> Proposal:
            return Proposal(edits={"playbook.md": "x"}, note="noop", kind="playbook")

        d2 = exp.step(noop)
        self.assertFalse(d2.accepted)

        self.assertIn("accepted", exp.history())
        self.assertIn("rejected", exp.history())

        exp.rollback()
        self.assertEqual(exp.current().read("prompt.md"), "weak")

        from simple_agent_lab.evolution.kernel import log

        rejected_hash = log.read(self.ws, accepted=False)[0].candidate["hash"]
        self.assertTrue((self.ws / "versions" / rejected_hash).is_dir())

    def test_from_config_builds_and_runs(self) -> None:
        from simple_agent_lab.evolution import Config, Use, registry
        from simple_agent_lab.evolution.types import Proposal, Run

        def make_rollout(**_):
            def rollout(version, slice_):
                reward = 0.7 if version.read("prompt.md") == "strong" else 0.3
                d = self.ws / "runs" / f"{version.hash}-{slice_.sha}" / "i1"
                (d / "out").mkdir(parents=True, exist_ok=True)
                (d / "out" / "result.json").write_text(json.dumps({"reward": reward}))
                return [Run(d)]

            return rollout

        registry.ROLLOUTS["demo_stub"] = make_rollout
        self.addCleanup(registry.ROLLOUTS.pop, "demo_stub", None)
        cfg = Config(
            workspace=self.ws,
            rollout=Use("demo_stub"),
            slice_id="demo",
            instances=({"instance_id": "i1"},),
            seed={"prompt.md": "weak"},
        )
        exp = Experiment.from_config(cfg)
        self.assertEqual(exp.current().read("prompt.md"), "weak")
        d = exp.step(lambda ctx: Proposal(edits={"prompt.md": "strong"}, kind="prompt"))
        self.assertIsNotNone(d)
        self.assertTrue(d.accepted)
        self.assertEqual(exp.current().read("prompt.md"), "strong")


if __name__ == "__main__":
    unittest.main()
