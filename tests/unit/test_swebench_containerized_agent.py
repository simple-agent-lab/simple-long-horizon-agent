from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from evals.swebench.containerized_agent import (
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
    _container_environment,
    build_runner_command,
    container_create_options,
    container_name,
    copy_file_to_container,
    copy_runner_support_files,
    load_instance as load_host_instance,
    prepare_wheelhouse,
    prepare_run_directory,
)
from evals.swebench.in_container_runner import (
    AGENT_SYSTEM_PROMPT,
    is_retryable_llm_error,
    load_instance as load_runner_instance,
    task_from_instance,
    with_llm_retry,
)


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

        self.assertIn("AGENT_PYTHON=/opt/miniconda3/bin/python3", command)
        self.assertIn(
            '"$AGENT_PYTHON" -m pip install --no-index --find-links /agent/wheelhouse',
            command,
        )
        self.assertIn("'simple-agent-lab[openai]'", command)
        self.assertIn(
            '"$AGENT_PYTHON" /agent/evals/swebench/in_container_runner.py',
            command,
        )
        self.assertIn("--workdir /testbed", command)
        self.assertNotIn("PYTHONPATH=", command)
        self.assertNotIn("/agent/simple-agent-lab", command)
        self.assertNotIn("docker exec", command)

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

        self.assertIn("instance_id: sympy__sympy-23824", task)
        self.assertIn("repo: sympy/sympy", task)
        self.assertIn("base_commit: abc123", task)
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
                "NO_PROXY": ".example.invalid",
                "no_proxy": ".internal.invalid",
            },
            clear=True,
        ):
            env = _container_environment("openai")

        self.assertEqual(env[OPENAI_MODEL_ENV], "gpt-test-1")
        self.assertEqual(env[OPENAI_AUTH_ENV], "token")
        self.assertEqual(env[OPENAI_BASE_URL_ENV], "https://example.invalid/v1")
        self.assertEqual(env["NO_PROXY"], ".example.invalid")
        self.assertEqual(env["no_proxy"], ".internal.invalid")

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
        self.assertIn("openai>=1.50.0", calls[1])

    def test_llm_retry_recovers_from_tpm_error_with_exponential_backoff(self) -> None:
        calls = 0
        sleeps: list[float] = []
        logs: list[str] = []
        expected = object()

        def flaky_step(agent, visible, state):
            nonlocal calls
            del agent, visible, state
            calls += 1
            if calls < 4:
                raise RuntimeError("TPM limit exceeded; retry after a while")
            return expected

        wrapped = with_llm_retry(
            flaky_step,
            sleep_fn=sleeps.append,
            log_fn=logs.append,
        )

        self.assertIs(wrapped(None, [], None), expected)
        self.assertEqual(calls, 4)
        self.assertEqual(sleeps, [4.0, 8.0, 16.0])
        self.assertEqual(len(logs), 3)
        self.assertIn("attempt 1/20", logs[0])
        self.assertIn("retrying in 4s", logs[0])

    def test_llm_retry_caps_delay_at_sixty_seconds(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def flaky_step(agent, visible, state):
            nonlocal calls
            del agent, visible, state
            calls += 1
            if calls < 8:
                raise RuntimeError("429 tokens per minute exceeded")
            return "ok"

        wrapped = with_llm_retry(
            flaky_step,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        self.assertEqual(wrapped(None, [], None), "ok")
        self.assertEqual(sleeps, [4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0])

    def test_llm_retry_does_not_retry_non_rate_limit_errors(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def broken_step(agent, visible, state):
            nonlocal calls
            del agent, visible, state
            calls += 1
            raise RuntimeError("invalid request body")

        wrapped = with_llm_retry(
            broken_step,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "invalid request body"):
            wrapped(None, [], None)
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])

    def test_llm_retry_raises_last_error_after_twenty_attempts(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def failing_step(agent, visible, state):
            nonlocal calls
            del agent, visible, state
            calls += 1
            raise RuntimeError(f"rate limit still active attempt={calls}")

        wrapped = with_llm_retry(
            failing_step,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "attempt=20"):
            wrapped(None, [], None)
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
