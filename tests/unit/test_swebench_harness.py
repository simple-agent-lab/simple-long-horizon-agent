from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.swebench.harness import (
    API_KIND_ENV,
    DEFAULT_MULTILINGUAL_RUN_ROOT,
    DEFAULT_MULTILINGUAL_WHEELHOUSE,
    DEFAULT_PRO_RUN_ROOT,
    DEFAULT_PRO_WHEELHOUSE,
    DEFAULT_RUN_ROOT,
    DEFAULT_WHEELHOUSE,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_SESSION_ID_ENV,
    _container_environment,
    _export_locked_requirements,
    _run_checked,
    container_entrypoint_override,
    docker_image_for_instance,
    docker_run_command,
    ensure_linux_uv,
    is_swebench_multilingual,
    is_swebench_pro,
    is_swebench_pro_instance,
    load_instance,
    prediction_record,
    prepare_project_wheel,
    prepare_wheelhouse,
    prepare_wheelhouse_for_run,
    resolve_api_kind,
    resolve_workdir,
    sanitized_instance,
)


def _record_command_with_fake_wheel(calls: list[list[str]], command: list[str]) -> None:
    calls.append(command)
    if command[:3] == ["uv", "build", "--wheel"]:
        out_dir = Path(command[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "simple_long_horizon_agent-0.1.0-py3-none-any.whl").write_bytes(
            b"wheel"
        )


class SwebenchHarnessTest(unittest.TestCase):
    """Host-side SWE-bench helpers shared by the suite, run entry, and scoring."""

    def test_default_artifact_paths_live_under_swebench_output_root(self) -> None:
        self.assertEqual(DEFAULT_RUN_ROOT, Path("evals/out/swebench").resolve())
        self.assertEqual(
            DEFAULT_WHEELHOUSE,
            Path("evals/out/swebench/wheelhouse/cp311-manylinux").resolve(),
        )
        self.assertEqual(
            DEFAULT_MULTILINGUAL_RUN_ROOT,
            Path("evals/out/swebench_multilingual").resolve(),
        )
        self.assertEqual(
            DEFAULT_MULTILINGUAL_WHEELHOUSE,
            Path(
                "evals/out/swebench_multilingual/wheelhouse/cp311-manylinux"
            ).resolve(),
        )
        self.assertEqual(DEFAULT_PRO_RUN_ROOT, Path("evals/out/swebench_pro").resolve())
        self.assertEqual(
            DEFAULT_PRO_WHEELHOUSE,
            Path("evals/out/swebench_pro/wheelhouse/cp311-manylinux").resolve(),
        )

    def test_linux_uv_is_downloaded_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"

            def download(_url: str, destination: Path) -> None:
                source = Path(tmp) / "uv"
                source.write_bytes(b"uv")
                with tarfile.open(destination, "w:gz") as bundle:
                    bundle.add(source, arcname="uv-test-target/uv")

            binary = ensure_linux_uv(
                target="uv-test-target", root=root, downloader=download
            )
            self.assertEqual(binary.read_bytes(), b"uv")
            self.assertTrue(binary.stat().st_mode & 0o100)

            with mock.patch(
                "evals.swebench.harness._download_file",
                side_effect=AssertionError("cache should be reused"),
            ):
                self.assertEqual(
                    ensure_linux_uv(target="uv-test-target", root=root), binary
                )

    def test_sanitized_instance_drops_gold_fields(self) -> None:
        instance = {
            "instance_id": "sympy__sympy-23824",
            "problem_statement": "fix it",
            "patch": "GOLD",
            "test_patch": "GOLD",
            "FAIL_TO_PASS": ["a"],
            "PASS_TO_PASS": ["b"],
        }
        sanitized = sanitized_instance(instance)
        self.assertEqual(
            sanitized,
            {"instance_id": "sympy__sympy-23824", "problem_statement": "fix it"},
        )

    def test_entrypoint_override_is_limited_to_pro_instances(self) -> None:
        self.assertEqual(
            container_entrypoint_override(
                {"instance_id": "sympy__sympy-23824"},
                dataset_name="princeton-nlp/SWE-bench_Verified",
            ),
            {},
        )
        self.assertEqual(
            container_entrypoint_override(
                {"instance_id": "instance_NodeBB__NodeBB-abc-vnan"},
                dataset_name="ScaleAI/SWE-bench_Pro",
            ),
            {"entrypoint": ""},
        )

    def test_pro_dataset_detection_only_accepts_hyphenated_marker(self) -> None:
        instance = {"instance_id": "sympy__sympy-23824"}

        self.assertTrue(
            is_swebench_pro_instance(instance, dataset_name="ScaleAI/SWE-bench_Pro")
        )
        self.assertTrue(
            is_swebench_pro(
                dataset_name="ScaleAI/SWE-bench_Pro",
                instance_id="sympy__sympy-23824",
            )
        )
        self.assertFalse(
            is_swebench_pro_instance(instance, dataset_name="local/swebench_pro")
        )
        self.assertFalse(
            is_swebench_pro(
                dataset_name="local/swebench_pro",
                instance_id="sympy__sympy-23824",
            )
        )

    def test_pro_instances_use_dockerhub_tag_image_and_app_workdir(self) -> None:
        instance = {
            "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
            "repo": "NodeBB/NodeBB",
            "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-abc",
        }

        self.assertEqual(
            docker_image_for_instance(
                instance,
                dataset_name="ScaleAI/SWE-bench_Pro",
                namespace="swebench",
                instance_image_tag="latest",
                env_image_tag="latest",
                dockerhub_username="jefzda",
            ),
            "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-abc",
        )
        self.assertEqual(
            resolve_workdir("", instance, dataset_name="ScaleAI/SWE-bench_Pro"),
            "/app",
        )
        self.assertEqual(
            docker_run_command(
                "echo ok", instance, dataset_name="ScaleAI/SWE-bench_Pro"
            ),
            ["/bin/sh", "-lc", "echo ok"],
        )

    def test_pro_image_falls_back_to_official_tag_shape(self) -> None:
        instance = {
            "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
            "repo": "NodeBB/NodeBB",
        }

        self.assertEqual(
            docker_image_for_instance(
                instance,
                dataset_name="ScaleAI/SWE-bench_Pro",
                namespace="swebench",
                instance_image_tag="latest",
                env_image_tag="latest",
                dockerhub_username="jefzda",
            ),
            "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-abc",
        )

    def test_verified_instances_keep_swebench_workdir(self) -> None:
        self.assertEqual(
            resolve_workdir(
                "",
                {"instance_id": "sympy__sympy-23824"},
                dataset_name="princeton-nlp/SWE-bench_Verified",
            ),
            "/testbed",
        )
        self.assertEqual(
            docker_run_command(
                "echo ok",
                {"instance_id": "sympy__sympy-23824"},
                dataset_name="princeton-nlp/SWE-bench_Verified",
            ),
            ["bash", "-lc", "echo ok"],
        )
        self.assertEqual(
            resolve_workdir(
                "/workspace",
                {"instance_id": "sympy__sympy-23824"},
                dataset_name="princeton-nlp/SWE-bench_Verified",
            ),
            "/workspace",
        )

    def test_verified_prediction_record_matches_official_eval_shape(self) -> None:
        record = prediction_record(
            "sympy__sympy-23824",
            "simple-long-horizon-agent-verified",
            "diff --git a/sympy/core.py b/sympy/core.py\n",
            dataset_name="princeton-nlp/SWE-bench_Verified",
        )

        self.assertEqual(record["instance_id"], "sympy__sympy-23824")
        self.assertEqual(
            record["model_name_or_path"], "simple-long-horizon-agent-verified"
        )
        self.assertEqual(
            record["model_patch"], "diff --git a/sympy/core.py b/sympy/core.py\n"
        )
        self.assertNotIn("prefix", record)
        self.assertNotIn("patch", record)

    def test_multilingual_prediction_record_matches_official_eval_shape(self) -> None:
        record = prediction_record(
            "kotlin__repo-123",
            "simple-long-horizon-agent-multilingual",
            "diff --git a/src/App.kt b/src/App.kt\n",
            dataset_name="SWE-bench/SWE-bench_Multilingual",
        )

        self.assertTrue(
            is_swebench_multilingual(dataset_name="SWE-bench/SWE-bench_Multilingual")
        )
        self.assertEqual(record["instance_id"], "kotlin__repo-123")
        self.assertEqual(
            record["model_name_or_path"], "simple-long-horizon-agent-multilingual"
        )
        self.assertEqual(
            record["model_patch"], "diff --git a/src/App.kt b/src/App.kt\n"
        )
        self.assertNotIn("prefix", record)
        self.assertNotIn("patch", record)

    def test_pro_prediction_record_matches_official_eval_shape(self) -> None:
        record = prediction_record(
            "instance_NodeBB__NodeBB-abc-vnan",
            "simple-long-horizon-agent-pro",
            "diff --git a/src/api.js b/src/api.js\n",
            dataset_name="ScaleAI/SWE-bench_Pro",
        )

        self.assertEqual(record["instance_id"], "instance_NodeBB__NodeBB-abc-vnan")
        self.assertEqual(record["prefix"], "simple-long-horizon-agent-pro")
        self.assertEqual(record["patch"], "diff --git a/src/api.js b/src/api.js\n")
        self.assertNotIn("model_patch", record)
        self.assertNotIn("model_name_or_path", record)

    def test_container_environment_passes_openai_env(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                OPENAI_MODEL_ENV: "gpt-test-1",
                OPENAI_AUTH_ENV: "token",
                OPENAI_BASE_URL_ENV: "https://example.invalid/v1",
                OPENAI_SESSION_ID_ENV: "session-1",
                OPENAI_LOG_ID_ENV: "log-1",
                API_KIND_ENV: "openai-responses",
                "NO_PROXY": ".example.invalid",
                "no_proxy": ".internal.invalid",
            },
            clear=True,
        ):
            env = _container_environment("openai")

        self.assertEqual(env[OPENAI_MODEL_ENV], "gpt-test-1")
        self.assertEqual(env[OPENAI_AUTH_ENV], "token")
        self.assertEqual(env[OPENAI_BASE_URL_ENV], "https://example.invalid/v1")
        self.assertEqual(env[OPENAI_SESSION_ID_ENV], "session-1")
        self.assertEqual(env[OPENAI_LOG_ID_ENV], "log-1")
        self.assertEqual(env[API_KIND_ENV], "openai-responses")
        self.assertEqual(env["NO_PROXY"], ".example.invalid")
        self.assertEqual(env["no_proxy"], ".internal.invalid")

    def test_container_environment_requires_model_and_auth(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(SystemExit, OPENAI_MODEL_ENV):
                _container_environment("openai")

    def test_resolve_api_kind_prefers_cli_then_env_then_chat_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_api_kind(None), "openai-chat")
            self.assertEqual(resolve_api_kind("openai-responses"), "openai-responses")

        with mock.patch.dict(
            "os.environ", {API_KIND_ENV: "openai-responses"}, clear=True
        ):
            self.assertEqual(resolve_api_kind(None), "openai-responses")
            self.assertEqual(resolve_api_kind("openai-chat"), "openai-chat")

    def test_resolve_api_kind_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Unsupported API_KIND"):
            resolve_api_kind("unknown-api")

    def test_load_instance_accepts_instances_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.json"
            path.write_text(
                json.dumps(
                    {
                        "instances": [
                            {"instance_id": "one", "problem_statement": "first"},
                            {"instance_id": "two", "problem_statement": "second"},
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            instance = load_instance(path, "two")

        self.assertEqual(instance["problem_statement"], "second")

    def test_prepare_wheelhouse_builds_package_and_downloads_provider_deps(
        self,
    ) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            _record_command_with_fake_wheel(calls, command)

        with tempfile.TemporaryDirectory() as tmp:
            prepare_wheelhouse(Path(tmp), runner=runner)
            requirements_path = str(Path(tmp) / "requirements.lock.txt")

        # Step 1: rebuild the project wheel.
        self.assertEqual(calls[0][:3], ["uv", "build", "--wheel"])
        # Step 2: export the locked runtime closure from uv.lock.
        self.assertEqual(Path(calls[1][0]).name, "uv")
        self.assertEqual(calls[1][1], "export")
        self.assertIn("--frozen", calls[1])
        self.assertIn("--no-dev", calls[1])
        self.assertIn("--no-emit-project", calls[1])
        self.assertEqual(calls[1][-2:], ["--output-file", requirements_path])
        # Steps 3+: pip download installs exactly the locked requirements.
        self.assertIn("download", calls[2])
        self.assertEqual(calls[2][-2:], ["-r", requirements_path])

    def test_prepare_wheelhouse_can_include_mcp_extra(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            _record_command_with_fake_wheel(calls, command)

        with tempfile.TemporaryDirectory() as tmp:
            prepare_wheelhouse(Path(tmp), runner=runner, extras=("mcp",))

        self.assertIn("--extra", calls[1])
        extra_index = calls[1].index("--extra")
        self.assertEqual(calls[1][extra_index + 1], "mcp")
        download_platforms = [
            command[command.index("--platform") + 1]
            for command in calls
            if "download" in command
        ]
        self.assertEqual(download_platforms, ["manylinux2014_x86_64"])

    def test_prepare_wheelhouse_provisions_offline_linux_cpython_311(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            _record_command_with_fake_wheel(calls, command)

        # Even on a macOS host (the common dev setup), the provisioned interpreter
        # targets the FIXED container platform (Linux x86_64 glibc), so the
        # wheelhouse carries the Python the Linux containers actually run.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("evals.swebench.harness.shutil.which", return_value="/fake/uv"),
            mock.patch("evals.swebench.harness.sys.platform", "darwin"),
        ):
            prepare_wheelhouse(Path(tmp), runner=runner)
            uv_python = str(Path(tmp) / "uv-python")
        self.assertEqual(
            calls[-1],
            [
                "/fake/uv",
                "python",
                "install",
                "--install-dir",
                uv_python,
                "cpython-3.11-linux-x86_64-gnu",
                "cpython-3.11-linux-x86_64-musl",
            ],
        )

    def test_prepare_wheelhouse_skips_python_provision_without_uv(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            _record_command_with_fake_wheel(calls, command)

        # No uv on PATH: nothing to provision with (the bootstrap then falls back
        # to the container's own download path).
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("evals.swebench.harness.shutil.which", return_value=None),
        ):
            prepare_wheelhouse(Path(tmp), runner=runner)
        self.assertFalse(any("install" in c and "python" in c for c in calls))

    def test_locked_requirements_pin_core_runtime_and_exclude_extra(self) -> None:
        uv = shutil.which("uv")
        if uv is None:
            self.skipTest("uv is not available on PATH")

        with tempfile.TemporaryDirectory() as tmp:
            requirements = _export_locked_requirements(uv, Path(tmp), run=_run_checked)
            text = requirements.read_text(encoding="utf-8")

        # Core runtime is pinned to exact versions (reproducible wheelhouse).
        self.assertRegex(text, r"(?m)^anthropic==")
        self.assertRegex(text, r"(?m)^openai==")
        for line in text.splitlines():
            requirement = line.split(";", 1)[0].strip()
            self.assertRegex(requirement, r"^[A-Za-z0-9_.-]+==[^<>=!~ ]+$")
        # Host-only dependencies (swebench extra + dev tools) stay out.
        self.assertNotRegex(text, r"(?m)^(datasets|docker|swebench|pytest|ruff)[=<>]")

    def test_prepare_project_wheel_refreshes_only_local_package(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            _record_command_with_fake_wheel(calls, command)

        with tempfile.TemporaryDirectory() as tmp:
            prepare_project_wheel(Path(tmp), runner=runner)
            wheel_names = sorted(path.name for path in Path(tmp).glob("*.whl"))

        self.assertEqual(calls[0][:4], ["uv", "build", "--wheel", "--out-dir"])
        self.assertNotEqual(calls[0][-1], tmp)
        self.assertEqual(
            wheel_names, ["simple_long_horizon_agent-0.1.0-py3-none-any.whl"]
        )

    def test_prepare_wheelhouse_for_run_refreshes_project_wheel_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp)
            with mock.patch(
                "evals.swebench.harness.prepare_project_wheel"
            ) as project_wheel:
                with mock.patch(
                    "evals.swebench.harness.prepare_wheelhouse"
                ) as full_wheelhouse:
                    prepare_wheelhouse_for_run(wheelhouse, prepare_all=False)

        project_wheel.assert_called_once_with(wheelhouse)
        full_wheelhouse.assert_not_called()

    def test_prepare_wheelhouse_for_run_can_prepare_full_wheelhouse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp)
            with mock.patch(
                "evals.swebench.harness.prepare_project_wheel"
            ) as project_wheel:
                with mock.patch(
                    "evals.swebench.harness.prepare_wheelhouse"
                ) as full_wheelhouse:
                    prepare_wheelhouse_for_run(wheelhouse, prepare_all=True)

        project_wheel.assert_not_called()
        full_wheelhouse.assert_called_once_with(wheelhouse)
