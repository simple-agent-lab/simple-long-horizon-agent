"""Bridge the trace-viewer's JS contract tests into the Python test run.

Defense 2 lives in JavaScript (studio/trace-viewer/test/*.test.mjs): it extracts
the viewer's real schema accessors from index.html and exercises them against the
generated fixture, so a producer/viewer schema drift fails instead of going
silently blank. This wrapper runs that node suite as part of
`python -m unittest discover`, so the one command the repo already uses covers it
too. It skips (does not fail) when node isn't available, so a Python-only
environment is unaffected.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[2] / "studio" / "trace-viewer"
# Lower bound on the node suite size. `node --test` exits 0 when it discovers
# NOTHING, so a green return code alone would be a false pass; assert a real
# count ran. Bump if you remove tests, never silently let it drop to 0.
MIN_NODE_TESTS = 5


class TraceViewerContractTest(unittest.TestCase):
    def test_node_contract_suite_passes(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH; run `npm test` in studio/trace-viewer")
        result = subprocess.run(
            [node, "--test"],
            cwd=VIEWER_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout + result.stderr

        def _count(label: str) -> int:
            match = re.search(rf"\b{label}\s+(\d+)\b", out)
            return int(match.group(1)) if match else -1

        ran, failed = _count("tests"), _count("fail")
        self.assertEqual(result.returncode, 0, f"node contract suite errored:\n{out}")
        self.assertEqual(failed, 0, f"trace-viewer contract failures:\n{out}")
        self.assertGreaterEqual(
            ran,
            MIN_NODE_TESTS,
            f"expected >= {MIN_NODE_TESTS} viewer contract tests but only {ran} ran "
            f"— did `node --test` discover studio/trace-viewer/test/?\n{out}",
        )
