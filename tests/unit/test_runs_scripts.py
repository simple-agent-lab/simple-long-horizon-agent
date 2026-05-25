from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RunsScriptsTest(unittest.TestCase):
    def test_swebench_run_scripts_have_valid_bash_syntax(self) -> None:
        scripts = [
            ROOT / "runs/eval_swebench.sh",
            ROOT / "runs/run_swebench_verified.sh",
            ROOT / "runs/run_swebench_pro.sh",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, result.stderr)

    def test_swebench_run_scripts_support_batch_flags(self) -> None:
        scripts = [
            ROOT / "runs/run_swebench_verified.sh",
            ROOT / "runs/run_swebench_pro.sh",
        ]

        for script in scripts:
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertIn("--all", text)
                self.assertIn("--parallel", text)
                self.assertIn("FETCH_PYTHON", text)
                self.assertIn("--with datasets", text)
                self.assertIn("wait -n", text)
                self.assertIn("prediction.jsonl", text)


if __name__ == "__main__":
    unittest.main()
