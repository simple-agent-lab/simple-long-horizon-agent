from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _load_run_swebench_suite_module():
    path = ROOT / "runs/_benches/swebench.py"
    spec = importlib.util.spec_from_file_location("sal_run_swebench_suite", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunsScriptsTest(unittest.TestCase):
    def test_env_example_includes_swebench_api_kind(self) -> None:
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("API_KIND=openai-chat", env_example)

    def test_swebench_extra_includes_dataset_fetch_dependencies(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"docker>=7.0.0"', pyproject)
        self.assertIn('"swebench>=3.0.0"', pyproject)
        self.assertIn('"datasets>=2.0.0"', pyproject)

    def test_swebench_run_scripts_have_valid_bash_syntax(self) -> None:
        scripts = [
            ROOT / "runs/eval_swebench.sh",
            ROOT / "runs/setup_swebench_docker.sh",
            ROOT / "runs/run_swebench_suite.sh",
            ROOT / "runs/run_swebench_gold_smoke.sh",
            ROOT / "runs/run_swebench_verified.sh",
            ROOT / "runs/run_swebench_multilingual.sh",
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
            ROOT / "runs/run_swebench_multilingual.sh",
            ROOT / "runs/run_swebench_pro.sh",
        ]

        for script in scripts:
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertIn("--all", text)
                self.assertIn("--parallel", text)
                self.assertIn("FETCH_PYTHON", text)
                self.assertIn("--extra swebench", text)
                self.assertIn("wait -n", text)
                self.assertIn("--collect-predictions", text)

    def test_swebench_run_scripts_load_provider_settings_from_dotenv(self) -> None:
        scripts = [
            ROOT / "runs/run_swebench_verified.sh",
            ROOT / "runs/run_swebench_multilingual.sh",
            ROOT / "runs/run_swebench_pro.sh",
        ]

        for script in scripts:
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertIn("--dotenv .env", text)

    def test_swebench_run_scripts_use_flat_suite_output_layout(self) -> None:
        expected_paths = {
            "run_swebench_verified.sh": [
                'INSTANCE_DIR="evals/out/swebench"',
                'CONTAINER_RUN_ROOT="evals/out/swebench"',
                'PREDICTION_DIR="evals/out/swebench"',
                "instance_${instance_id}.jsonl",
            ],
            "run_swebench_pro.sh": [
                'INSTANCE_DIR="evals/out/swebench_pro"',
                'CONTAINER_RUN_ROOT="evals/out/swebench_pro"',
                'PREDICTION_DIR="evals/out/swebench_pro"',
                'WHEELHOUSE="evals/out/swebench_pro/wheelhouse/cp311-manylinux"',
                "instance_${instance_id}.jsonl",
                '--wheelhouse "$WHEELHOUSE"',
            ],
            "run_swebench_multilingual.sh": [
                'INSTANCE_DIR="evals/out/swebench_multilingual"',
                'CONTAINER_RUN_ROOT="evals/out/swebench_multilingual"',
                'PREDICTION_DIR="evals/out/swebench_multilingual"',
                'WHEELHOUSE="evals/out/swebench_multilingual/wheelhouse/cp311-manylinux"',
                "instance_${instance_id}.jsonl",
                '--wheelhouse "$WHEELHOUSE"',
            ],
            "setup_swebench_docker.sh": [
                "evals/out/swebench/instance_${INSTANCE_ID}.jsonl",
                "evals/out/swebench/wheelhouse",
            ],
            "run_swebench_gold_smoke.sh": [
                "evals/out/swebench_official",
            ],
            "eval_swebench.sh": [
                "evals/out/swebench_predictions.jsonl",
                "evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl",
                "evals/out/swebench_pro/swebench_pro_predictions.jsonl",
            ],
        }

        for script_name, paths in expected_paths.items():
            text = (ROOT / "runs" / script_name).read_text(encoding="utf-8")
            with self.subTest(script=script_name):
                for path in paths:
                    self.assertIn(path, text)
                self.assertNotIn("evals/out/swebench/verified", text)
                self.assertNotIn("evals/out/swebench/pro", text)
                self.assertNotIn("evals/out/swebench/shared", text)

    def test_swebench_suite_entry_uses_multilingual_default_paths(self) -> None:
        module = _load_run_swebench_suite_module()
        args = Namespace(
            dataset_name="SWE-bench/SWE-bench_Multilingual",
            run_root=None,
            wheelhouse=None,
        )

        run_root, wheelhouse = module._resolve_paths(
            args, {"instance_id": "kotlin__repo-123"}
        )

        self.assertEqual(run_root, ROOT / "evals/out/swebench_multilingual")
        self.assertEqual(
            wheelhouse,
            ROOT / "evals/out/swebench_multilingual/wheelhouse/cp311-manylinux",
        )

    def test_swebench_output_docs_show_flat_suite_layout(self) -> None:
        docs = [
            ROOT / "evals/out/README.md",
            ROOT / "evals/out/swebench/README.md",
            ROOT / "evals/out/swebench_multilingual/README.md",
            ROOT / "evals/out/swebench_pro/README.md",
        ]

        for path in docs:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("instance_<id>.jsonl", text)
                self.assertIn("wheelhouse/", text)
                self.assertIn("<run-id>/", text)
                self.assertNotIn("├── verified/", text)
                self.assertNotIn("├── pro/", text)
                self.assertNotIn("└── shared/", text)

    def test_verified_swebench_entries_do_not_use_lite_dataset(self) -> None:
        files = [
            ROOT / "runs/setup_swebench_docker.sh",
            ROOT / "runs/run_swebench_suite.sh",
            ROOT / "runs/run_swebench_gold_smoke.sh",
            ROOT / "evals/swebench/evaluate_predictions.py",
            ROOT / "evals/swebench/README.md",
        ]

        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn("SWE-bench_Lite", text)


if __name__ == "__main__":
    unittest.main()
