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


def _run(
    script: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", f"runs/{script}", *args],
        cwd=ROOT,
        env=env,
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
        self.assertIn("--config", result.stdout)
        self.assertIn("configs/dgm_swebench.yaml", result.stdout)
        self.assertIn("--train-dataset", result.stdout)
        self.assertIn("--test-dataset", result.stdout)
        self.assertIn("--execute", result.stdout)
        self.assertIn("--parent-selection", result.stdout)
        self.assertIn("--monitor", result.stdout)
        self.assertIn("--branches", result.stdout)
        self.assertIn("--parallel", result.stdout)
        self.assertIn("Global Docker worker cap.", result.stdout)
        self.assertNotIn("auto", result.stdout.lower())

    def test_dgm_wrapper_rejects_auto_parallel(self):
        result = _run(
            "run_dgm_swebench.sh",
            "--parallel",
            "auto",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a positive integer", result.stderr)

    def test_dgm_wrapper_invokes_recipe_as_module(self):
        script = (ROOT / "runs" / "run_dgm_swebench.sh").read_text(encoding="utf-8")

        self.assertIn("-m recipes.dgm.evolve", script)
        self.assertNotIn("recipes/dgm/evolve.py", script)

    def test_dgm_monitor_uses_ops_report_path(self):
        source = (ROOT / "recipes" / "dgm" / "evolve.py").read_text(encoding="utf-8")

        self.assertIn('"ops" / "report.py"', source)
        self.assertNotIn('"dgm" / "report.py"', source)

    def test_simple_wrapper_prepares_docker_and_linux_uv_for_execute(self):
        script = (ROOT / "runs" / "run_self_evolving_simple.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("source runs/_swebench_uv.sh", script)
        self.assertIn("source runs/_docker.sh", script)
        self.assertIn("docker_ensure_running", script)
        self.assertIn("swebench_ensure_linux_uv", script)


if __name__ == "__main__":
    unittest.main()
