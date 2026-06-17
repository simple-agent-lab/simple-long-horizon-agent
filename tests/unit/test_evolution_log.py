from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import log
from simple_agent_lab.evolution.types import Slice, Verdict


class LogTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)

    def _append(self, accepted: bool, kind: str) -> None:
        log.append(
            self.ws,
            baseline={"hash": "aaa", "scores": {"reward": 0.3}},
            candidate={"hash": "bbb", "parent": "aaa", "scores": {"reward": 0.5}},
            slice_=Slice("demo", ({"instance_id": "i1"},)),
            verdict=Verdict(accepted, "because", {"reward": 0.2}),
            kind=kind,
            runs={"baseline": "r-a", "candidate": "r-b"},
        )

    def test_append_assigns_id_and_reads_back(self) -> None:
        self._append(True, "prompt")
        self._append(False, "lesson")
        rows = log.read(self.ws)
        self.assertEqual([d.id for d in rows], ["d-000001", "d-000002"])
        self.assertEqual(rows[0].accepted, True)
        self.assertEqual(rows[0].kind, "prompt")
        self.assertEqual(rows[0].deltas["reward"], 0.2)
        self.assertEqual(rows[0].runs["candidate"], "r-b")

    def test_read_filters_and_hit_rate(self) -> None:
        self._append(True, "prompt")
        self._append(False, "prompt")
        self._append(True, "prompt")
        self.assertEqual(len(log.read(self.ws, kind="prompt")), 3)
        self.assertAlmostEqual(log.hit_rate(self.ws, kind="prompt"), 2 / 3)


if __name__ == "__main__":
    unittest.main()
