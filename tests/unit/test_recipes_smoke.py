import importlib.util
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


if __name__ == "__main__":
    unittest.main()
