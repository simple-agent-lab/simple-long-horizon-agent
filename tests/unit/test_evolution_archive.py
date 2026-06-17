from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from recipes.dgm import archive
from simple_agent_lab.evolution.kernel import log
from simple_agent_lab.evolution.types import Slice, Verdict


class ArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)

    def _decision(
        self,
        *,
        baseline: str,
        candidate: str,
        parent: str | None,
        score: float,
        valid_parent: bool = True,
    ) -> None:
        log.append(
            self.ws,
            baseline={"hash": baseline, "scores": {"reward": score - 0.1}},
            candidate={
                "hash": candidate,
                "parent": parent,
                "scores": {"reward": score},
                "valid_parent": valid_parent,
            },
            slice_=Slice("demo", ({"instance_id": "i1"},)),
            verdict=Verdict(score > 0.0, "scored", {"reward": 0.1}),
            kind="code",
            runs={"baseline": f"run-{baseline}", "candidate": f"run-{candidate}"},
        )

    def test_nodes_reconstruct_scores_lineage_and_validity(self) -> None:
        self._decision(baseline="root", candidate="a", parent="root", score=0.3)
        self._decision(baseline="a", candidate="b", parent="a", score=0.7)
        self._decision(
            baseline="a", candidate="bad", parent="a", score=0.9, valid_parent=False
        )

        nodes = archive.nodes(self.ws)
        by_hash = {node.hash: node for node in nodes}

        self.assertEqual([node.hash for node in nodes], ["root", "a", "b", "bad"])
        self.assertEqual(by_hash["b"].parent, "a")
        self.assertEqual(by_hash["b"].scores["reward"], 0.7)
        self.assertFalse(by_hash["bad"].valid_parent)

    def test_select_parent_filters_invalid_and_supports_common_methods(self) -> None:
        self._decision(baseline="root", candidate="a", parent="root", score=0.3)
        self._decision(baseline="a", candidate="b", parent="a", score=0.7)
        self._decision(
            baseline="a", candidate="bad", parent="a", score=0.9, valid_parent=False
        )

        nodes = archive.nodes(self.ws)

        self.assertEqual(archive.select_parent(nodes, method="latest"), "b")
        self.assertEqual(archive.select_parent(nodes, method="best"), "b")
        self.assertEqual(
            archive.select_parent(
                nodes, method="score_child_prop", rng=random.Random(0)
            ),
            "b",
        )


if __name__ == "__main__":
    unittest.main()
