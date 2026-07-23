from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.unit._support import load_module


ROOT = Path(__file__).resolve().parents[2]


def _load_run_swebench_suite_module():
    return load_module(ROOT / "runs/_benches/swebench.py", "sal_run_swebench_suite")


def _load_run_programbench_suite_module():
    return load_module(
        ROOT / "runs/_benches/programbench.py", "sal_run_programbench_suite"
    )


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
        module = _load_run_swebench_suite_module()
        args = module._build_batch_parser().parse_args(
            ["--all", "--parallel", "3", "--pull"]
        )

        self.assertTrue(args.all)
        self.assertEqual(args.parallel, 3)
        self.assertEqual(args.pull, "missing")

    def test_swebench_run_scripts_load_provider_settings_from_dotenv(self) -> None:
        module = _load_run_swebench_suite_module()
        args = module._build_batch_parser().parse_args([])
        with (
            mock.patch.object(module.harness, "load_dotenv") as load_dotenv,
            mock.patch.object(
                module.harness,
                "_container_environment",
                return_value={"OPENAI_MODEL": "model", "OPENAI_AUTH_TOKEN": "token"},
            ),
        ):
            module._provider_environment(args)

        load_dotenv.assert_called_once_with(str(ROOT / ".env"))

    def test_swebench_variants_cover_all_three_splits(self) -> None:
        """The one runner parametrizes all three splits via --variant."""
        variants = _load_run_swebench_suite_module().VARIANTS
        self.assertEqual(set(variants), {"verified", "multilingual", "pro"})
        self.assertEqual(
            {variant.dataset for variant in variants.values()},
            {
                "princeton-nlp/SWE-bench_Verified",
                "SWE-bench/SWE-bench_Multilingual",
                "ScaleAI/SWE-bench_Pro",
            },
        )

    def test_swebench_run_scripts_use_flat_suite_output_layout(self) -> None:
        variants = _load_run_swebench_suite_module().VARIANTS
        self.assertEqual(
            {variant.run_root for variant in variants.values()},
            {
                ROOT / "evals/out/swebench",
                ROOT / "evals/out/swebench_multilingual",
                ROOT / "evals/out/swebench_pro",
            },
        )
        self.assertEqual(
            variants["pro"].wheelhouse,
            ROOT / "evals/out/swebench_pro/wheelhouse/cp311-manylinux",
        )
        self.assertEqual(
            variants["multilingual"].wheelhouse,
            ROOT / "evals/out/swebench_multilingual/wheelhouse/cp311-manylinux",
        )
        expected_paths = {
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

    def test_swebench_ids_keep_file_order_and_reject_duplicates(self) -> None:
        module = _load_run_swebench_suite_module()
        rows = [{"instance_id": "one"}, {"instance_id": "two"}]
        with tempfile.TemporaryDirectory() as tmp:
            ids = Path(tmp) / "ids.txt"
            ids.write_text("two\n# note\none\n", encoding="utf-8")
            args = module._build_batch_parser().parse_args(["--ids-file", str(ids)])
            with mock.patch.object(module, "_dataset_rows", return_value=rows):
                selected = module._select_instances(args, module.VARIANTS["verified"])
            self.assertEqual([row["instance_id"] for row in selected], ["two", "one"])

            ids.write_text("one\none\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "Duplicate"):
                module._ids_from_file(ids)

    def test_swebench_batch_prepares_once_and_alternates_tokens(self) -> None:
        module = _load_run_swebench_suite_module()
        rows = [{"instance_id": value} for value in ("one", "two", "three")]
        report = SimpleNamespace(summary=lambda: {"total": 3, "ok": 3, "failed": 0})
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp) / "wheels"
            args = module._build_batch_parser().parse_args(
                [
                    "--all",
                    "--run-root",
                    tmp,
                    "--wheelhouse",
                    str(wheelhouse),
                    "--uv-binary",
                    "/tmp/uv",
                    "--run-id",
                    "unsafe/id",
                ]
            )
            with (
                mock.patch.dict("os.environ", {"OPENAI_AUTH_TOKEN2": "secondary"}),
                mock.patch.object(module, "_select_instances", return_value=rows),
                mock.patch.object(
                    module,
                    "_provider_environment",
                    return_value={
                        module.harness.API_KIND_ENV: "openai-chat",
                        module.harness.OPENAI_AUTH_ENV: "primary",
                    },
                ),
                mock.patch.object(
                    module, "_suite", return_value=SimpleNamespace(name="swebench")
                ),
                mock.patch.object(module.docker_cli, "backend", return_value=object()),
                mock.patch.object(
                    module, "run_container_batch", return_value=(report, 0)
                ) as run_batch,
                mock.patch.object(module.harness, "prepare_wheelhouse") as prepare,
                mock.patch(
                    "evals.swebench.evaluate_predictions.predictions_from_run_dirs",
                    return_value=[],
                ),
            ):
                module.run_batch(args)

            prepare.assert_called_once_with(wheelhouse.resolve())
            self.assertEqual(
                run_batch.call_args.kwargs["run_id"],
                module.canonical_run_id("unsafe/id"),
            )
            per_instance = run_batch.call_args.kwargs["per_instance_kwargs"]
            tokens = [
                per_instance(row)["provider_env"][module.harness.OPENAI_AUTH_ENV]
                for row in rows
            ]
            self.assertEqual(tokens, ["primary", "secondary", "primary"])

    def test_programbench_batch_keeps_multi_ids_and_filter_slice(self) -> None:
        module = _load_run_programbench_suite_module()
        parser = module._build_batch_parser()
        args = parser.parse_args(["one", "two", "--parallel", "2", "--pull"])
        with mock.patch.object(
            module.harness,
            "load_instance",
            side_effect=lambda value: {"instance_id": value},
        ):
            rows = module._batch_instances(args)
        self.assertEqual([row["instance_id"] for row in rows], ["one", "two"])
        self.assertEqual(args.pull, "missing")

        args = parser.parse_args(["--filter", "org.*", "--slice", "0:5"])
        with mock.patch.object(
            module.harness,
            "load_instances",
            return_value=[{"instance_id": "filtered"}],
        ) as load:
            self.assertEqual(
                module._batch_instances(args)[0]["instance_id"], "filtered"
            )
        load.assert_called_once_with(filter_spec="org.*", slice_spec="0:5")

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
            "harbor",
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
