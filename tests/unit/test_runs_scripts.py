from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]

# The per-dataset SWE-bench launchers — thin wrappers that set SWEBENCH_*
# constants and delegate to the shared driver runs/_swebench_run.sh.
SWEBENCH_LAUNCHERS = (
    "run_swebench_verified.sh",
    "run_swebench_multilingual.sh",
    "run_swebench_pro.sh",
)


def _load_run_swebench_suite_module():
    path = ROOT / "runs/run_swebench_suite.py"
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
            ROOT / "runs/_swebench_run.sh",
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
        # The batch machinery lives once in the shared driver; each thin
        # launcher routes into it via `source` + `swebench_main`.
        driver = (ROOT / "runs/_swebench_run.sh").read_text(encoding="utf-8")
        for token in (
            "--all",
            "--parallel",
            "FETCH_PYTHON",
            "--extra swebench",
            "wait -n",
            "--collect-predictions",
        ):
            self.assertIn(token, driver)

        for name in SWEBENCH_LAUNCHERS:
            text = (ROOT / "runs" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("source runs/_swebench_run.sh", text)
                self.assertIn('swebench_main "$@"', text)

    def test_swebench_run_scripts_load_provider_settings_from_dotenv(self) -> None:
        # `run_container` (in the shared driver) passes `--dotenv .env` for
        # every dataset launcher.
        driver = (ROOT / "runs/_swebench_run.sh").read_text(encoding="utf-8")
        self.assertIn("--dotenv .env", driver)

    def test_swebench_run_scripts_use_flat_suite_output_layout(self) -> None:
        # Each launcher names its suite root once; the shared driver derives the
        # flat instance/wheelhouse/run-dir layout from SWEBENCH_RUN_ROOT.
        launcher_run_roots = {
            "run_swebench_verified.sh": 'SWEBENCH_RUN_ROOT="evals/out/swebench"',
            "run_swebench_pro.sh": 'SWEBENCH_RUN_ROOT="evals/out/swebench_pro"',
            "run_swebench_multilingual.sh": (
                'SWEBENCH_RUN_ROOT="evals/out/swebench_multilingual"'
            ),
        }
        for script_name, run_root in launcher_run_roots.items():
            text = (ROOT / "runs" / script_name).read_text(encoding="utf-8")
            with self.subTest(script=script_name):
                self.assertIn(run_root, text)
                self.assertNotIn("evals/out/swebench/verified", text)
                self.assertNotIn("evals/out/swebench/pro", text)
                self.assertNotIn("evals/out/swebench/shared", text)

        driver = (ROOT / "runs/_swebench_run.sh").read_text(encoding="utf-8")
        for token in (
            "${SWEBENCH_RUN_ROOT}/instance_${instance_id}.jsonl",
            "${SWEBENCH_RUN_ROOT}/wheelhouse/cp311-manylinux",
            "${SWEBENCH_RUN_ROOT}/${RUN_ID}/",
        ):
            self.assertIn(token, driver)

        other_paths = {
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
        for script_name, paths in other_paths.items():
            text = (ROOT / "runs" / script_name).read_text(encoding="utf-8")
            with self.subTest(script=script_name):
                for path in paths:
                    self.assertIn(path, text)

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
