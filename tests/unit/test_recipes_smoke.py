import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimpleRecipeSmokeTest(unittest.TestCase):
    def test_simple_parser_builds_and_is_dry_by_default(self):
        mod = _load(ROOT / "recipes" / "simple" / "evolve.py")
        args = mod.build_parser().parse_args(
            ["--train-dataset", "t.jsonl", "--test-dataset", "e.jsonl", "--run-id", "x"]
        )
        self.assertFalse(args.execute)
        self.assertEqual(args.rounds, 4)


class DgmRecipeSmokeTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load(ROOT / "recipes" / "dgm" / "evolve.py")

    def test_parser_exposes_dgm_knobs(self):
        ns = self.mod.build_parser().parse_args(
            [
                "--run-id",
                "x",
                "--train-dataset",
                "t.jsonl",
                "--test-dataset",
                "e.jsonl",
                "--parent-selection",
                "best",
                "--branches",
                "4",
            ]
        )
        self.assertEqual(ns.parent_selection, "best")
        self.assertEqual(ns.branches, 4)
        self.assertFalse(ns.execute)

    def test_pick_best_node_selects_highest_valid(self):
        from simple_agent_lab.evolution import archive

        nodes = (
            archive.ArchiveNode(hash="a", scores={"reward": 0.5}),
            archive.ArchiveNode(hash="b", scores={"reward": 0.9}),
            archive.ArchiveNode(hash="d", scores={"reward": 1.0}, valid_parent=False),
            archive.ArchiveNode(hash="e", scores={}),
        )
        best = self.mod.pick_best_node(nodes)
        self.assertIsNotNone(best)
        self.assertEqual(best.hash, "b")

    def test_pick_best_node_none_when_empty(self):
        self.assertIsNone(self.mod.pick_best_node(()))

    def test_score_reads_reward_from_candidate(self):
        self.assertEqual(self.mod._score({"scores": {"reward": 0.5}}), 0.5)
        self.assertEqual(self.mod._score({}), 0.0)
        self.assertEqual(self.mod._score({"scores": "bad"}), 0.0)

    def test_heldout_run_id_uses_version_and_slice(self):
        from simple_agent_lab.evolution.types import Version

        with tempfile.TemporaryDirectory() as tmp:
            vd = Path(tmp) / "abc123"
            (vd / "agent").mkdir(parents=True)
            (vd / "agent" / "agent_program.py").write_text(
                "def build_agent(): ...\n", encoding="utf-8"
            )
            rid = self.mod.heldout_run_id(Version(vd), ({"instance_id": "test-1"},))
        self.assertTrue(rid.startswith("abc123-"))


if __name__ == "__main__":
    unittest.main()
