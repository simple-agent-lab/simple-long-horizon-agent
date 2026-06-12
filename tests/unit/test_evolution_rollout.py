from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import RESULT_KEY, TRACE_KEY, FakeBackend, LocalDirStore
from simple_agent_lab.evals.protocols import LaunchSpec, RunSpec
from simple_agent_lab.evolution.components.rollout import dataset_rollout
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Slice


class _DemoSuite:
    name = "demo"
    container_module = "demo.container"

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        return LaunchSpec(image="demo:latest", workdir="/work")

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def eval_inputs(self, instance: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None


def _simulate(reward: float):
    def on_run(spec: RunSpec, bound) -> None:
        bound.put(TRACE_KEY, b'{"events": []}\n')
        bound.put(RESULT_KEY, (json.dumps({"reward": reward}) + "\n").encode("utf-8"))

    return on_run


class RolloutTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name).resolve()
        self.version = store.stage(self.ws, base=None, edits={"prompt.md": "hi"})

    def test_runs_one_per_instance_and_reads_reward(self) -> None:
        rollout = dataset_rollout(
            suite=_DemoSuite(),
            backend=FakeBackend(on_run=_simulate(0.5)),
            store=LocalDirStore(self.ws / "runs"),
            runs_root=self.ws / "runs",
        )
        slice_ = Slice("demo", ({"instance_id": "i1"}, {"instance_id": "i2"}))
        runs = rollout(self.version, slice_)
        self.assertEqual(len(runs), 2)
        self.assertEqual({r.instance_id for r in runs}, {"i1", "i2"})
        self.assertTrue(all(r.reward == 0.5 for r in runs))

    def test_reuses_existing_run_dir(self) -> None:
        backend = FakeBackend(on_run=_simulate(0.5))
        rollout = dataset_rollout(
            suite=_DemoSuite(),
            backend=backend,
            store=LocalDirStore(self.ws / "runs"),
            runs_root=self.ws / "runs",
        )
        slice_ = Slice("demo", ({"instance_id": "i1"},))
        first = rollout(self.version, slice_)
        run_id = first[0].run_id
        again = rollout(self.version, slice_)  # same (version, slice) -> reuse
        self.assertEqual(again[0].run_id, run_id)

    def test_partial_prior_run_is_rerolled(self) -> None:
        slice_ = Slice("demo", ({"instance_id": "i1"},))
        run_id = f"{self.version.hash}-{slice_.sha}"
        # Simulate a crashed prior run: the instance dir exists (the harness
        # creates it at run START) but never wrote out/result.json.
        ((self.ws / "runs" / run_id / "i1").mkdir(parents=True))
        rollout = dataset_rollout(
            suite=_DemoSuite(),
            backend=FakeBackend(on_run=_simulate(0.5)),
            store=LocalDirStore(self.ws / "runs"),
            runs_root=self.ws / "runs",
        )
        runs = rollout(self.version, slice_)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].instance_id, "i1")
        self.assertEqual(runs[0].reward, 0.5)


if __name__ == "__main__":
    unittest.main()
