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

    def test_env_example_does_not_advertise_local_second_openai_token(self) -> None:
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertNotIn("OPENAI_AUTH_TOKEN2", env_example)

    def test_swebench_extra_includes_dataset_fetch_dependencies(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"docker>=7.0.0"', pyproject)
        self.assertIn('"swebench>=3.0.0"', pyproject)
        self.assertIn('"datasets>=2.0.0"', pyproject)

    def test_swebench_run_scripts_have_valid_bash_syntax(self) -> None:
        scripts = [
            ROOT / "runs/swebench/eval_swebench.sh",
            ROOT / "runs/swebench/setup_swebench_docker.sh",
            ROOT / "runs/swebench/run_swebench_gold_smoke.sh",
            ROOT / "runs/swebench/run_swebench.sh",
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
        text = (ROOT / "runs/swebench/run_swebench.sh").read_text(encoding="utf-8")
        self.assertIn("--all", text)
        self.assertIn("--parallel", text)
        self.assertIn("FETCH_PYTHON", text)
        self.assertIn("--extra swebench", text)
        self.assertIn("wait -n", text)
        self.assertIn("--collect-predictions", text)

    def test_swebench_run_scripts_load_provider_settings_from_dotenv(self) -> None:
        text = (ROOT / "runs/swebench/run_swebench.sh").read_text(encoding="utf-8")
        self.assertIn("--dotenv .env", text)

    def test_swebench_run_script_can_round_robin_second_openai_token(self) -> None:
        text = (ROOT / "runs/swebench/run_swebench.sh").read_text(encoding="utf-8")

        self.assertIn("OPENAI_AUTH_TOKEN2", text)
        self.assertIn("SECONDARY_OPENAI_AUTH_TOKEN", text)
        self.assertIn('OPENAI_AUTH_TOKEN="$SECONDARY_OPENAI_AUTH_TOKEN"', text)
        self.assertIn("run_container_for_index", text)

    def test_swebench_variants_cover_all_three_splits(self) -> None:
        """The one runner parametrizes all three splits via --variant."""
        text = (ROOT / "runs/swebench/run_swebench.sh").read_text(encoding="utf-8")
        for variant in ("verified", "multilingual", "pro"):
            self.assertIn(variant, text, f"missing --variant case: {variant}")
        self.assertIn("princeton-nlp/SWE-bench_Verified", text)
        self.assertIn("SWE-bench/SWE-bench_Multilingual", text)
        self.assertIn("ScaleAI/SWE-bench_Pro", text)

    def test_swebench_pro_defaults_to_250_agent_turns(self) -> None:
        text = (ROOT / "runs/swebench/run_swebench.sh").read_text(encoding="utf-8")
        pro_case = text.split("  pro)\n", maxsplit=1)[1].split("  *)", maxsplit=1)[0]

        self.assertIn("MAX_TURNS=250", pro_case)

    def test_swebench_run_scripts_use_flat_suite_output_layout(self) -> None:
        expected_paths = {
            "run_swebench.sh": [
                'OUT_ROOT="evals/out/swebench"',
                'OUT_ROOT="evals/out/swebench_multilingual"',
                'OUT_ROOT="evals/out/swebench_pro"',
                'WHEELHOUSE="evals/out/swebench_pro/wheelhouse/cp311-manylinux"',
                'WHEELHOUSE="evals/out/swebench_multilingual/wheelhouse/cp311-manylinux"',
                'INSTANCE_DIR="$OUT_ROOT"',
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
            text = (ROOT / "runs/swebench" / script_name).read_text(encoding="utf-8")
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
            ROOT / "runs/swebench/setup_swebench_docker.sh",
            ROOT / "runs/swebench/run_swebench_gold_smoke.sh",
            ROOT / "evals/swebench/evaluate_predictions.py",
            ROOT / "evals/swebench/README.md",
        ]

        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn("SWE-bench_Lite", text)

    def test_runs_toplevel_is_whitelisted(self) -> None:
        """`runs/` top level stays tidy: only the public entry + known subdirs.

        Guards the reorg — a new ad-hoc script must go into a concern subdir
        (swebench/ programbench/ demos/ dev/), the internal _benches/, or the
        profiles/ lib/ data dirs, not loose at the top.
        """
        allowed_files = {
            "run_bench.py",
            "README.md",
            "bench-manifest.example.json",
        }
        allowed_dirs = {
            "_benches",
            "profiles",
            "lib",
            "swebench",
            "programbench",
            "demos",
            "dev",
        }
        for entry in (ROOT / "runs").iterdir():
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            with self.subTest(entry=entry.name):
                if entry.is_dir():
                    self.assertIn(
                        entry.name,
                        allowed_dirs,
                        f"unexpected top-level dir runs/{entry.name}/ — add it to "
                        "allowed_dirs only if it's a real concern group.",
                    )
                else:
                    self.assertIn(
                        entry.name,
                        allowed_files,
                        f"unexpected top-level file runs/{entry.name} — the only "
                        "public entry is run_bench.py; per-bench logic goes in "
                        "runs/_benches/ and scripts in a concern subdir.",
                    )


if __name__ == "__main__":
    unittest.main()
