from __future__ import annotations

import inspect
import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from evals.swebench import in_container_runner
from evals.swebench.containerized_agent import (
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_SESSION_ID_ENV,
    _container_environment,
    _ensure_image_available,
    build_runner_command,
    container_create_options,
    container_entrypoint_override,
    container_name,
    copy_file_to_container,
    copy_runner_support_files,
    docker_image_for_instance,
    docker_run_command,
    load_instance as load_host_instance,
    prepare_project_wheel,
    prepare_wheelhouse_for_run,
    prepare_wheelhouse,
    prepare_run_directory,
    resolve_api_kind,
    resolve_workdir,
)
from evals.swebench.in_container_runner import (
    AGENT_SYSTEM_PROMPT,
    build_openai_request_extra_from_env,
    build_openai_provider_from_env,
    is_retryable_llm_error,
    load_instance as load_runner_instance,
    prediction_record,
    run_agent,
    task_from_instance,
    trace_from_state,
    with_llm_retry,
)
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab import make_llm_agent
from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.state import State
from simple_agent_lab.trajectory import trace_record


class SwebenchContainerizedAgentTest(unittest.TestCase):
    def test_container_name_is_stable_and_docker_safe(self) -> None:
        self.assertEqual(
            container_name("sympy__sympy-23824", "container/run:1"),
            "sweb.agent.sympy__sympy-23824.container_run_1",
        )

    def test_container_create_options_omits_empty_network_mode(self) -> None:
        self.assertEqual(container_create_options(""), {})
        self.assertEqual(container_create_options(None), {})

    def test_container_create_options_sets_network_mode(self) -> None:
        self.assertEqual(container_create_options("host"), {"network_mode": "host"})

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

    def test_runner_command_installs_agent_and_runs_inside_container(self) -> None:
        command = build_runner_command(
            run_mount="/agent/run",
            instance_id="sympy__sympy-23824",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            model_name="simple-agent-lab-mimo-v2.5-pro",
            provider="openai",
            max_turns=20,
            wheelhouse_mount="/agent/wheelhouse",
        )

        self.assertIn("set -eu", command)
        self.assertIn("set -o pipefail", command)
        self.assertIn("UV_BIN=", command)
        self.assertIn("[ -f /tmp/uv ]", command)
        self.assertIn("/tmp/uv --version", command)
        self.assertIn("command -v uv", command)
        self.assertIn('"$PATH_UV" --version', command)
        self.assertIn('"$UV_BIN" venv --python 3.11 /tmp/agent-venv', command)
        self.assertNotIn("miniconda", command.lower())
        self.assertNotIn("conda create", command)
        self.assertIn("python3 -m venv /tmp/agent-venv", command)
        self.assertIn("_IS_MUSL", command)
        self.assertIn("venv", command)
        self.assertIn(
            '"$UV_BIN" pip install --python "$AGENT_PYTHON" --no-index --find-links /agent/wheelhouse',
            command,
        )
        self.assertIn(
            '"$AGENT_PYTHON" -m pip install --no-index --find-links /agent/wheelhouse',
            command,
        )
        self.assertNotIn("--break-system-packages", command)
        self.assertIn(" simple-agent-lab", command)
        self.assertNotIn("simple-agent-lab[openai]", command)
        self.assertIn(
            '"$AGENT_PYTHON" /agent/evals/swebench/in_container_runner.py',
            command,
        )
        self.assertIn("--workdir /testbed", command)
        self.assertIn("--api-kind openai-chat", command)
        self.assertNotIn("PYTHONPATH=", command)
        self.assertNotIn("/agent/simple-agent-lab", command)
        self.assertNotIn("docker exec", command)

    def test_runner_command_can_select_openai_responses(self) -> None:
        command = build_runner_command(
            run_mount="/agent/run",
            instance_id="sympy__sympy-23824",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            model_name="simple-agent-lab-responses",
            provider="openai",
            max_turns=20,
            api_kind="openai-responses",
        )

        self.assertIn("--api-kind openai-responses", command)

    def test_task_tells_agent_it_is_inside_container(self) -> None:
        task = task_from_instance(
            {
                "instance_id": "sympy__sympy-23824",
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Fix gamma matrices.",
            },
            workdir="/testbed",
        )

        self.assertNotIn("instance_id: sympy__sympy-23824", task)
        self.assertNotIn("repo: sympy/sympy", task)
        self.assertNotIn("base_commit: abc123", task)
        self.assertIn("You are running inside the SWE-bench container.", task)
        self.assertIn("The bash tool runs locally in /testbed.", task)
        self.assertIn("Fix gamma matrices.", task)
        self.assertIn("Recommended workflow:", task)
        self.assertIn("Read relevant files before editing.", task)
        self.assertIn("Create or run a small reproduction", task)
        self.assertIn(
            "Do not modify tests, reproduction files, or configuration files unless",
            task,
        )
        self.assertIn("do not include a patch", task)
        self.assertNotIn("docker exec", task)

    def test_runner_can_thread_request_extra_through_agent_preset(self) -> None:
        self.assertIn("request_extra", inspect.signature(run_agent).parameters)
        self.assertIn("request_extra", inspect.signature(make_bash_agent).parameters)
        self.assertIn("request_extra", inspect.signature(make_llm_agent).parameters)

    def test_pro_task_includes_requirements_and_interface(self) -> None:
        task = task_from_instance(
            {
                "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
                "repo": "NodeBB/NodeBB",
                "base_commit": "abc123",
                "problem_statement": "Add the moderation endpoint.",
                "requirements": "The endpoint must require administrator access.",
                "interface": "POST /api/v3/moderation/queue accepts JSON.",
            },
            workdir="/app",
        )

        self.assertIn("The bash tool runs locally in /app.", task)
        self.assertIn("problem_statement:", task)
        self.assertIn("Add the moderation endpoint.", task)
        self.assertIn("requirements:", task)
        self.assertIn("administrator access", task)
        self.assertIn("interface:", task)
        self.assertIn("POST /api/v3/moderation/queue", task)

    def test_pro_task_omits_empty_optional_context_sections(self) -> None:
        task = task_from_instance(
            {
                "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
                "problem_statement": "Fix a route.",
                "requirements": None,
                "interface": "",
            },
            workdir="/app",
        )

        self.assertNotIn("requirements:", task)
        self.assertNotIn("interface:", task)

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
                "echo ok",
                instance,
                dataset_name="ScaleAI/SWE-bench_Pro",
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

    def test_pro_prediction_record_matches_official_eval_shape(self) -> None:
        record = prediction_record(
            "instance_NodeBB__NodeBB-abc-vnan",
            "simple-agent-lab-pro",
            "diff --git a/src/api.js b/src/api.js\n",
            dataset_name="ScaleAI/SWE-bench_Pro",
        )

        self.assertEqual(record["instance_id"], "instance_NodeBB__NodeBB-abc-vnan")
        self.assertEqual(record["prefix"], "simple-agent-lab-pro")
        self.assertEqual(record["patch"], "diff --git a/src/api.js b/src/api.js\n")
        self.assertNotIn("model_patch", record)
        self.assertNotIn("model_name_or_path", record)

    def test_pro_trace_meta_uses_pro_suite_name(self) -> None:
        state = State(
            task="Solve this SWE-bench instance.",
            data={"model_patch": "diff --git a/a b/a\n", "workspace": "/app"},
        )

        trace = trace_from_state(
            state=state,
            instance={"instance_id": "instance_NodeBB__NodeBB-abc-vnan"},
            dataset_name="ScaleAI/SWE-bench_Pro",
            split="test",
            model_name="model",
            patch_source="containerized-diff",
        )

        assert trace.meta is not None
        self.assertEqual(trace.meta["suite"], "swebench_pro")

    def test_run_agent_sets_pro_suite_in_state_data(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.generate = lambda *args: None

            def run(self, task, max_turns):
                del task, max_turns
                return State(task="Solve this SWE-bench instance."), []

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                in_container_runner,
                "prepare_baseline_commit",
                return_value="base",
            ),
            mock.patch.object(
                in_container_runner,
                "make_bash_agent",
                return_value=FakeAgent(),
            ),
            mock.patch.object(
                in_container_runner,
                "git_diff",
                return_value="diff --git a/a b/a\n",
            ),
        ):
            state = run_agent(
                instance={
                    "instance_id": "instance_NodeBB__NodeBB-abc-vnan",
                    "problem_statement": "Fix a route.",
                },
                provider=Provider(id="fake", api="fake", model="fake-model"),
                workdir=Path(tmp),
                max_turns=1,
                dataset_name="ScaleAI/SWE-bench_Pro",
            )

        self.assertEqual(state.data["suite"], "swebench_pro")

    def test_trace_from_state_serializes_event_kind(self) -> None:
        state = State(task="Solve this SWE-bench instance.")
        state.send("task", sender="user", target="swebench_agent", content="Fix it.")

        trace = trace_from_state(
            state=state,
            instance={"instance_id": "sympy__sympy-23824"},
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            model_name="model",
            patch_source="containerized-diff",
        )

        record = trace_record(trace)

        self.assertEqual(record["events"][0]["kind"], "message")
        self.assertIn("message", record["events"][0])

    def test_system_prompt_sets_swebench_repair_operating_rules(self) -> None:
        self.assertIn("general and consistent with the codebase", AGENT_SYSTEM_PROMPT)
        self.assertIn("Each bash tool call runs in a fresh shell", AGENT_SYSTEM_PROMPT)
        self.assertIn("Use non-interactive command flags", AGENT_SYSTEM_PROMPT)
        self.assertIn("Keep command output focused", AGENT_SYSTEM_PROMPT)
        self.assertNotIn("parallel bash tool calls", AGENT_SYSTEM_PROMPT)

    def test_copy_file_to_container_writes_runner_archive(self) -> None:
        class FakeContainer:
            def __init__(self) -> None:
                self.path = ""
                self.data = b""

            def put_archive(self, path: str, data: bytes) -> None:
                self.path = path
                self.data = data

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "runner.py"
            source.write_text("print('ok')\n", encoding="utf-8")
            container = FakeContainer()

            copy_file_to_container(
                container,
                source_path=source,
                target_path="/agent/evals/swebench/in_container_runner.py",
            )

            self.assertEqual(container.path, "/")
            with tarfile.open(fileobj=BytesIO(container.data), mode="r") as archive:
                names = archive.getnames()
                member = archive.extractfile(
                    "agent/evals/swebench/in_container_runner.py"
                )
                self.assertIsNotNone(member)
                content = member.read().decode("utf-8") if member else ""

        self.assertIn("agent/evals/swebench", names)
        self.assertEqual(content, "print('ok')\n")

    def test_copy_runner_support_files_copies_eval_dependencies(self) -> None:
        class FakeContainer:
            def __init__(self) -> None:
                self.targets: list[str] = []

            def put_archive(self, path: str, data: bytes) -> None:
                with tarfile.open(fileobj=BytesIO(data), mode="r") as archive:
                    self.targets.extend(archive.getnames())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "in_container_runner.py"
            runner.write_text("# runner\n", encoding="utf-8")
            (root / "patch_extract.py").write_text(
                "# patch extract\n", encoding="utf-8"
            )
            container = FakeContainer()

            copy_runner_support_files(
                container,
                runner_path=runner,
                container_runner_path="/agent/evals/swebench/in_container_runner.py",
            )

        self.assertIn("agent/evals/swebench/patch_extract.py", container.targets)

    def test_container_environment_passes_openai_env(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                OPENAI_MODEL_ENV: "gpt-test-1",
                OPENAI_AUTH_ENV: "token",
                OPENAI_BASE_URL_ENV: "https://example.invalid/v1",
                OPENAI_SESSION_ID_ENV: "session-1",
                OPENAI_LOG_ID_ENV: "log-1",
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
        self.assertEqual(env["NO_PROXY"], ".example.invalid")
        self.assertEqual(env["no_proxy"], ".internal.invalid")

    def test_openai_request_extra_from_env_builds_responses_headers(self) -> None:
        extra = build_openai_request_extra_from_env(
            env={
                OPENAI_SESSION_ID_ENV: "session-1",
                OPENAI_LOG_ID_ENV: "log-1",
            }
        )

        self.assertEqual(extra["extra_headers"]["X-TT-logid"], "log-1")
        self.assertEqual(
            json.loads(extra["extra_headers"]["extra"]),
            {"session_id": "session-1"},
        )

    def test_openai_request_extra_from_env_requires_header_pair(self) -> None:
        with self.assertRaisesRegex(SystemExit, OPENAI_LOG_ID_ENV):
            build_openai_request_extra_from_env(
                env={
                    OPENAI_SESSION_ID_ENV: "session-1",
                }
            )

    def test_resolve_api_kind_prefers_cli_then_env_then_chat_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_api_kind(None), "openai-chat")
            self.assertEqual(resolve_api_kind("openai-responses"), "openai-responses")

        with mock.patch.dict(
            "os.environ",
            {API_KIND_ENV: "openai-responses"},
            clear=True,
        ):
            self.assertEqual(resolve_api_kind(None), "openai-responses")
            self.assertEqual(resolve_api_kind("openai-chat"), "openai-chat")

    def test_resolve_api_kind_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Unsupported API_KIND"):
            resolve_api_kind("unknown-api")

    def test_openai_provider_from_env_selects_responses_api(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                OPENAI_MODEL_ENV: "gpt-test-1",
                OPENAI_AUTH_ENV: "token",
                OPENAI_BASE_URL_ENV: "https://example.invalid/v1",
            },
            clear=True,
        ):
            provider = build_openai_provider_from_env("openai-responses")

        self.assertEqual(provider.id, "openai-responses")
        self.assertEqual(provider.api, "openai-responses")
        self.assertEqual(provider.model, "gpt-test-1")
        self.assertEqual(provider.base_url, "https://example.invalid/v1")
        self.assertEqual(provider.api_key_env, OPENAI_AUTH_ENV)
        self.assertGreaterEqual(provider.default_max_tokens or 0, 32768)

    def test_task_decodes_json_encoded_pro_context_fields(self) -> None:
        task = task_from_instance(
            {
                "problem_statement": json.dumps("Line one\nLine two"),
                "requirements": json.dumps("- Requirement A\n- Requirement B"),
                "interface": json.dumps("No new interfaces are introduced."),
            },
            workdir="/app",
        )

        self.assertIn("problem_statement:\nLine one\nLine two", task)
        self.assertIn("requirements:\n- Requirement A\n- Requirement B", task)
        self.assertIn("interface:\nNo new interfaces are introduced.", task)
        self.assertNotIn('"Line one\\nLine two"', task)

    def test_prepare_run_directory_writes_sanitized_instance_input(self) -> None:
        instance = {
            "instance_id": "sympy__sympy-23824",
            "repo": "sympy/sympy",
            "base_commit": "abc123",
            "problem_statement": "Fix it.",
            "patch": "gold patch",
            "test_patch": "gold tests",
            "FAIL_TO_PASS": ["secret"],
            "PASS_TO_PASS": ["secret"],
            "selected_test_files_to_run": ["secret_test.py"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_run_directory(
                run_root=Path(tmp),
                instance=instance,
                run_id="container-run",
            )
            record = json.loads(paths.instance_json.read_text(encoding="utf-8"))

        self.assertEqual(record["instance_id"], "sympy__sympy-23824")
        self.assertEqual(record["problem_statement"], "Fix it.")
        self.assertNotIn("patch", record)
        self.assertNotIn("test_patch", record)
        self.assertNotIn("FAIL_TO_PASS", record)
        self.assertNotIn("PASS_TO_PASS", record)
        self.assertNotIn("selected_test_files_to_run", record)

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

            host_instance = load_host_instance(path, "two")
            runner_instance = load_runner_instance(path, "one")

        self.assertEqual(host_instance["problem_statement"], "second")
        self.assertEqual(runner_instance["problem_statement"], "first")

    def test_prepare_wheelhouse_builds_package_and_downloads_provider_deps(
        self,
    ) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            calls.append(command)

        with tempfile.TemporaryDirectory() as tmp:
            prepare_wheelhouse(Path(tmp), runner=runner)

        self.assertEqual(calls[0][:3], ["uv", "build", "--wheel"])
        self.assertIn("anthropic>=0.39.0", calls[1])
        self.assertIn("openai>=1.50.0", calls[1])

    def test_pull_always_refreshes_existing_image(self) -> None:
        class FakeImages:
            def __init__(self) -> None:
                self.gets: list[str] = []
                self.pulls: list[tuple[str, str | None]] = []

            def get(self, image_key: str) -> object:
                self.gets.append(image_key)
                return object()

            def pull(self, image_key: str, platform: str | None = None) -> None:
                self.pulls.append((image_key, platform))

        images = FakeImages()

        _ensure_image_available(
            images,
            image_key="example/image:tag",
            platform="linux/amd64",
            pull_policy="always",
            image_not_found_error=LookupError,
            docker_exception=RuntimeError,
        )

        self.assertEqual(images.gets, ["example/image:tag"])
        self.assertEqual(images.pulls, [("example/image:tag", "linux/amd64")])

    def test_pull_missing_uses_existing_image_without_refresh(self) -> None:
        class FakeImages:
            def __init__(self) -> None:
                self.pulls: list[tuple[str, str | None]] = []

            def get(self, image_key: str) -> object:
                del image_key
                return object()

            def pull(self, image_key: str, platform: str | None = None) -> None:
                self.pulls.append((image_key, platform))

        images = FakeImages()

        _ensure_image_available(
            images,
            image_key="example/image:tag",
            platform=None,
            pull_policy="missing",
            image_not_found_error=LookupError,
            docker_exception=RuntimeError,
        )

        self.assertEqual(images.pulls, [])

    def test_prepare_project_wheel_refreshes_only_local_package(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str]) -> None:
            calls.append(command)

        with tempfile.TemporaryDirectory() as tmp:
            prepare_project_wheel(Path(tmp), runner=runner)

        self.assertEqual(calls, [["uv", "build", "--wheel", "--out-dir", tmp]])

    def test_prepare_wheelhouse_for_run_refreshes_project_wheel_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp)
            with mock.patch(
                "evals.swebench.containerized_agent.prepare_project_wheel"
            ) as project_wheel:
                with mock.patch(
                    "evals.swebench.containerized_agent.prepare_wheelhouse"
                ) as full_wheelhouse:
                    prepare_wheelhouse_for_run(wheelhouse, prepare_all=False)

        project_wheel.assert_called_once_with(wheelhouse)
        full_wheelhouse.assert_not_called()

    def test_prepare_wheelhouse_for_run_can_prepare_full_wheelhouse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp)
            with mock.patch(
                "evals.swebench.containerized_agent.prepare_project_wheel"
            ) as project_wheel:
                with mock.patch(
                    "evals.swebench.containerized_agent.prepare_wheelhouse"
                ) as full_wheelhouse:
                    prepare_wheelhouse_for_run(wheelhouse, prepare_all=True)

        project_wheel.assert_not_called()
        full_wheelhouse.assert_called_once_with(wheelhouse)

    def test_llm_retry_recovers_from_tpm_error_with_exponential_backoff(self) -> None:
        calls = 0
        sleeps: list[float] = []
        logs: list[str] = []
        expected = object()

        def flaky_generate(visible):
            nonlocal calls
            del visible
            calls += 1
            if calls < 4:
                raise RuntimeError("TPM limit exceeded; retry after a while")
            return expected

        wrapped = with_llm_retry(
            flaky_generate,
            sleep_fn=sleeps.append,
            log_fn=logs.append,
        )

        self.assertIs(wrapped([]), expected)
        self.assertEqual(calls, 4)
        self.assertEqual(sleeps, [4.0, 8.0, 16.0])
        self.assertEqual(len(logs), 3)
        self.assertIn("attempt 1/20", logs[0])
        self.assertIn("retrying in 4s", logs[0])

    def test_llm_retry_caps_delay_at_sixty_seconds(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def flaky_generate(visible):
            nonlocal calls
            del visible
            calls += 1
            if calls < 8:
                raise RuntimeError("429 tokens per minute exceeded")
            return "ok"

        wrapped = with_llm_retry(
            flaky_generate,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        self.assertEqual(wrapped([]), "ok")
        self.assertEqual(sleeps, [4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0])

    def test_llm_retry_does_not_retry_non_rate_limit_errors(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def broken_generate(visible):
            nonlocal calls
            del visible
            calls += 1
            raise RuntimeError("invalid request body")

        wrapped = with_llm_retry(
            broken_generate,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "invalid request body"):
            wrapped([])
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])

    def test_llm_retry_raises_last_error_after_twenty_attempts(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def failing_generate(visible):
            nonlocal calls
            del visible
            calls += 1
            raise RuntimeError(f"rate limit still active attempt={calls}")

        wrapped = with_llm_retry(
            failing_generate,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "attempt=20"):
            wrapped([])
        self.assertEqual(calls, 20)
        self.assertEqual(len(sleeps), 19)
        self.assertEqual(sleeps[:5], [4.0, 8.0, 16.0, 32.0, 60.0])
        self.assertEqual(sleeps[-1], 60.0)

    def test_retryable_llm_error_matches_common_tpm_and_rate_limit_text(self) -> None:
        self.assertTrue(is_retryable_llm_error(RuntimeError("TPM exceeded")))
        self.assertTrue(
            is_retryable_llm_error(RuntimeError("tokens per minute exhausted"))
        )
        self.assertTrue(is_retryable_llm_error(RuntimeError("HTTP 429")))
        self.assertTrue(is_retryable_llm_error(RuntimeError("Too Many Requests")))
        self.assertFalse(is_retryable_llm_error(RuntimeError("invalid schema")))
