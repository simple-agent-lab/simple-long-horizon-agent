from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.types import Run, Slice, Version


class TypesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_slice_sha_is_stable_and_order_independent(self) -> None:
        a = Slice("s", ({"instance_id": "b"}, {"instance_id": "a"}))
        b = Slice("s", ({"instance_id": "a"}, {"instance_id": "b"}))
        self.assertEqual(a.sha, b.sha)
        self.assertEqual(len(a.sha), 12)

    def test_run_reads_result_and_reward(self) -> None:
        run_dir = self.tmp / "r1" / "i1"
        (run_dir / "out").mkdir(parents=True)
        (run_dir / "out" / "result.json").write_text(json.dumps({"reward": 0.5}))
        run = Run(run_dir)
        self.assertTrue(run.ok)
        self.assertEqual(run.instance_id, "i1")
        self.assertEqual(run.reward, 0.5)
        self.assertEqual(run.result["reward"], 0.5)

    def test_run_missing_result_is_not_ok(self) -> None:
        run_dir = self.tmp / "r1" / "i2"
        run_dir.mkdir(parents=True)
        run = Run(run_dir)
        self.assertFalse(run.ok)
        self.assertIsNone(run.reward)

    def test_version_reads_files(self) -> None:
        vdir = self.tmp / "versions" / "abc"
        vdir.mkdir(parents=True)
        (vdir / "prompt.md").write_text("hello")
        v = Version(vdir)
        self.assertEqual(v.hash, "abc")
        self.assertEqual(v.read("prompt.md"), "hello")
        self.assertEqual(v.read("missing.md"), "")


if __name__ == "__main__":
    unittest.main()
