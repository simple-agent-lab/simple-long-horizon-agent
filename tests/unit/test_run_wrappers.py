from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _help(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", f"runs/{script}", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class RunWrapperHelpTest(unittest.TestCase):
    def test_simple_wrapper_help_documents_execution_inputs(self):
        result = _help("run_self_evolving_simple.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        usage = next(
            line for line in result.stdout.splitlines() if line.startswith("usage:")
        )
        self.assertIn("--config", result.stdout)
        self.assertIn("[--config CONFIG]", usage)
        self.assertIn("configs/simple_swebench.yaml", result.stdout)
        self.assertIn("--run-id", result.stdout)
        self.assertIn("--execute", result.stdout)
        self.assertIn("--reset", result.stdout)
        self.assertIn("--monitor", result.stdout)

    def test_dgm_wrapper_help_documents_knobs(self):
        result = _help("run_dgm_swebench.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--train-dataset", result.stdout)
        self.assertIn("--test-dataset", result.stdout)
        self.assertIn("--execute", result.stdout)
        self.assertIn("--parent-selection", result.stdout)
        self.assertIn("--monitor", result.stdout)


if __name__ == "__main__":
    unittest.main()
