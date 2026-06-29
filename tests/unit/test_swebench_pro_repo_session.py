from __future__ import annotations

import argparse
import base64
import os
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

from simple_agent_lab import Agent, message_text, run
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import (
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    is_tool_result_message,
    message_tool_calls,
    text_of,
    tool_results_message,
    user_message,
)
from simple_agent_lab.protocols import ContextCompressionEvent, TurnStartEvent
from simple_agent_lab.tools import tool_result_text


class SwebenchProRepoSessionPlanningTest(unittest.TestCase):
    def test_groups_by_repo_and_sorts_by_commit_timestamp(self) -> None:
        from evals.swebench.pro_repo_session import (
            group_instances_by_repo,
            sort_repo_instances,
        )

        rows = [
            {"instance_id": "b", "repo": "acme/widgets", "base_commit": "sha-b"},
            {"instance_id": "a", "repo": "acme/widgets", "base_commit": "sha-a"},
            {"instance_id": "c", "repo": "other/pkg", "base_commit": "sha-c"},
        ]

        groups = group_instances_by_repo(rows)
        ordered = sort_repo_instances(
            groups["acme/widgets"],
            commit_times={"sha-a": 100, "sha-b": 200},
        )

        self.assertEqual(sorted(groups), ["acme/widgets", "other/pkg"])
        self.assertEqual([row["instance_id"] for row in ordered], ["a", "b"])

    def test_unknown_commit_times_fall_back_to_dataset_order_after_known_commits(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import sort_repo_instances

        rows = [
            {"instance_id": "first", "base_commit": "missing-1"},
            {"instance_id": "known", "base_commit": "known"},
            {"instance_id": "second", "base_commit": "missing-2"},
        ]

        ordered = sort_repo_instances(rows, commit_times={"known": 10})

        self.assertEqual(
            [row["instance_id"] for row in ordered],
            ["known", "first", "second"],
        )

    def test_repo_session_part_count_uses_long_tail_thresholds(self) -> None:
        from evals.swebench.pro_repo_session import repo_session_part_count

        self.assertEqual(repo_session_part_count(20), 1)
        self.assertEqual(repo_session_part_count(50), 1)
        self.assertEqual(repo_session_part_count(51), 2)
        self.assertEqual(repo_session_part_count(80), 2)
        self.assertEqual(repo_session_part_count(81), 3)

    def test_split_repo_session_parts_preserves_order_in_near_even_chunks(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import split_repo_session_parts

        rows = [
            {"instance_id": f"case-{index:03d}", "repo": "acme/widgets"}
            for index in range(85)
        ]

        parts = split_repo_session_parts("acme/widgets", rows)

        self.assertEqual([part.part_index for part in parts], [1, 2, 3])
        self.assertEqual([part.part_count for part in parts], [3, 3, 3])
        self.assertEqual([len(part.rows) for part in parts], [29, 28, 28])
        self.assertEqual(parts[0].rows[0]["instance_id"], "case-000")
        self.assertEqual(parts[1].rows[0]["instance_id"], "case-029")
        self.assertEqual(parts[2].rows[-1]["instance_id"], "case-084")

    def test_current_pro_dataset_counts_resolve_to_23_session_parts(self) -> None:
        from evals.swebench.pro_repo_session import repo_session_part_count

        repo_counts = {
            "NodeBB/NodeBB": 44,
            "ansible/ansible": 96,
            "element-hq/element-web": 56,
            "flipt-io/flipt": 85,
            "future-architect/vuls": 62,
            "gravitational/teleport": 76,
            "internetarchive/openlibrary": 91,
            "navidrome/navidrome": 57,
            "protonmail/WebClients": 65,
            "qutebrowser/qutebrowser": 79,
            "tutao/tutanota": 20,
        }

        self.assertEqual(
            sum(repo_session_part_count(n) for n in repo_counts.values()), 23
        )

    def test_default_experiment_config_matches_requested_compression_setup(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import ProRepoExperimentConfig

        config = ProRepoExperimentConfig()

        self.assertEqual(config.model, "")
        self.assertEqual(config.api_kind, "openai-responses")
        self.assertEqual(config.reasoning_effort, "")
        self.assertEqual(config.threshold_tokens, 217_600)
        self.assertEqual(config.keep_recent, 12)
        self.assertEqual(config.max_turns, 250)
        self.assertEqual(config.agent_flavor, "bash")
        self.assertEqual(config.compression_strategy, "summarize")
        self.assertEqual(
            config.preserve_kinds, ("task", "system", "context", "summary")
        )

    def test_build_summarize_policy_uses_same_provider_and_records_parameters(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import ProRepoExperimentConfig

        provider = Provider(
            id="openai-chat",
            api="openai-chat",
            model="gpt-5.4-2026-03-05",
            api_key_env="OPENAI_AUTH_TOKEN",
            default_reasoning="high",
        )
        config = ProRepoExperimentConfig()

        policy = config.context_policy(provider, request_extra={"x": "y"})

        strategy = policy.strategy
        self.assertIsInstance(strategy, SummarizeStrategy)
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 217_600)
        self.assertEqual(strategy.keep_recent, 12)
        self.assertEqual(
            strategy.preserve_kinds, ("task", "system", "context", "summary")
        )
        self.assertEqual(strategy.compressor.name, "swebench_compressor")
        self.assertEqual(strategy.compressor.context_policy, None)

    def test_repo_session_runner_defaults_to_openai_responses_api_kind(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertEqual(args.api_kind, "openai-responses")

    def test_repo_session_runner_leaves_model_and_reasoning_to_env_by_default(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertIsNone(args.model)
        self.assertIsNone(args.reasoning_effort)
        self.assertIsNone(args.provider_auth_envs)

    def test_provider_auth_envs_expand_to_one_slot_per_session(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _expand_provider_auth_envs,
        )

        auth_envs = _expand_provider_auth_envs(
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11",
            session_count=23,
        )

        self.assertEqual(auth_envs[:12], ["OPENAI_AUTH_TOKEN"] * 12)
        self.assertEqual(auth_envs[12:], ["OPENAI_AUTH_TOKEN2"] * 11)

    def test_provider_auth_envs_use_primary_token_by_default(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _expand_provider_auth_envs,
        )

        self.assertEqual(
            _expand_provider_auth_envs(None, session_count=3),
            ["OPENAI_AUTH_TOKEN", "OPENAI_AUTH_TOKEN", "OPENAI_AUTH_TOKEN"],
        )

    def test_provider_slots_use_api_kind_temperature_defaults(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _providers_from_auth_envs,
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "model",
                "OPENAI_AUTH_TOKEN": "token-1",
                "OPENAI_AUTH_TOKEN2": "token-2",
                "REASONING_EFFORT": "high",
            },
            clear=True,
        ):
            responses = _providers_from_auth_envs(
                ["OPENAI_AUTH_TOKEN", "OPENAI_AUTH_TOKEN2"],
                api_kind="openai-responses",
            )
            chat = _providers_from_auth_envs(
                ["OPENAI_AUTH_TOKEN"],
                api_kind="openai-chat",
            )

        self.assertIsNone(responses["OPENAI_AUTH_TOKEN"].default_temperature)
        self.assertIsNone(responses["OPENAI_AUTH_TOKEN2"].default_temperature)
        self.assertEqual(chat["OPENAI_AUTH_TOKEN"].default_temperature, 1.0)
        self.assertEqual(responses["OPENAI_AUTH_TOKEN"].default_reasoning, "high")
        self.assertEqual(
            responses["OPENAI_AUTH_TOKEN2"].api_key_env, "OPENAI_AUTH_TOKEN2"
        )

    def test_provider_auth_envs_fail_when_slots_do_not_cover_sessions(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _expand_provider_auth_envs,
        )

        with self.assertRaises(SystemExit) as raised:
            _expand_provider_auth_envs("OPENAI_AUTH_TOKEN:1", session_count=2)

        self.assertIn("provides 1 auth slot", str(raised.exception))

    def test_experiment_config_records_effective_env_model_and_reasoning(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _apply_provider_env_overrides,
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])

        with patch.dict(
            os.environ,
            {"OPENAI_MODEL": "env-model", "REASONING_EFFORT": "medium"},
            clear=True,
        ):
            _apply_provider_env_overrides(args)
            config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.model, "env-model")
        self.assertEqual(config.reasoning_effort, "medium")

    def test_repo_session_runner_preserves_prior_summaries(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])

        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(
            config.preserve_kinds, ("task", "system", "context", "summary")
        )

    def test_responses_request_extra_includes_encrypted_reasoning(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _request_extra_for_api_kind,
        )

        extra = _request_extra_for_api_kind("openai-responses")

        self.assertEqual(extra["include"], ["reasoning.encrypted_content"])

    def test_non_responses_request_extra_does_not_include_encrypted_reasoning(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _request_extra_for_api_kind,
        )

        extra = _request_extra_for_api_kind("openai-chat")

        self.assertNotIn("include", extra)

    def test_cli_model_and_reasoning_override_env_when_explicit(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _apply_provider_env_overrides,
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--all",
                "--model",
                "cli-model",
                "--reasoning-effort",
                "low",
            ]
        )

        with patch.dict(
            os.environ,
            {"OPENAI_MODEL": "env-model", "REASONING_EFFORT": "medium"},
            clear=True,
        ):
            _apply_provider_env_overrides(args)
            config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.model, "cli-model")
        self.assertEqual(config.reasoning_effort, "low")


class DockerExecBashToolTest(unittest.TestCase):
    def test_docker_runner_executes_command_with_standard_bash_shell(self) -> None:
        from evals.swebench.pro_repo_session import DockerCommandRunner

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        runner = DockerCommandRunner(run=run)

        result = runner.exec(
            container_name="sal.instance",
            workdir="/app",
            command="pwd",
            timeout_seconds=13,
        )

        self.assertEqual(
            calls[0],
            [
                "docker",
                "exec",
                "-w",
                "/app",
                "sal.instance",
                "bash",
                "-lc",
                "pwd",
            ],
        )
        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[0], kwargs_command(calls[0]))

    def test_container_bash_tool_model_contract_matches_standard_bash_tool(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )
        from simple_agent_lab.tools.bash import make_bash_tool

        tool = make_container_bash_tool(
            CurrentContainer(), DockerCommandRunner(run=lambda *a, **k: None)
        )
        standard = make_bash_tool(cwd="/app")

        self.assertEqual(tool.name, standard.name)
        self.assertEqual(tool.description, standard.description)
        self.assertEqual(tool.parameters, standard.parameters)
        self.assertEqual(tool.execution_mode, standard.execution_mode)
        self.assertEqual(tool.timeout_seconds, standard.timeout_seconds)

    def test_docker_runner_starts_pro_container_for_long_lived_session(self) -> None:
        from evals.swebench.pro_repo_session import DockerCommandRunner

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="container-id\n", stderr="")

        runner = DockerCommandRunner(run=run)

        runner.start_container(
            container_name="sal.instance",
            image="jefzda/sweap-images:repo.commit",
            workdir="/app",
            network_mode="host",
            mem_limit="8g",
        )

        self.assertEqual(
            calls[0],
            [
                "docker",
                "run",
                "-d",
                "--name",
                "sal.instance",
                "--entrypoint",
                "",
                "--network",
                "host",
                "--memory",
                "8g",
                "-w",
                "/app",
                "jefzda/sweap-images:repo.commit",
                "/bin/sh",
                "-lc",
                "sleep infinity",
            ],
        )

    def test_docker_runner_removes_container_forcefully(self) -> None:
        from evals.swebench.pro_repo_session import DockerCommandRunner

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        runner = DockerCommandRunner(run=run)

        runner.remove_container("sal.instance")

        self.assertEqual(calls[0], ["docker", "rm", "-f", "sal.instance"])

    def test_container_bash_tool_reports_missing_current_container(self) -> None:
        from evals.swebench.pro_repo_session import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )

        tool = make_container_bash_tool(
            CurrentContainer(), DockerCommandRunner(run=lambda *a, **k: None)
        )

        result = tool.execute("call", {"command": "pwd"}, lambda: False, None)

        self.assertTrue(result.is_error)
        self.assertIn("No active SWE-bench container", text_of(result.content))

    def test_container_bash_tool_blocks_long_leading_sleep_like_standard_tool(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            del kwargs
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        current = CurrentContainer(name="sal.instance", workdir="/app")
        tool = make_container_bash_tool(current, DockerCommandRunner(run=run))

        result = tool.execute(
            "call",
            {"command": "sleep 2", "description": "wait"},
            lambda: False,
            None,
        )

        self.assertTrue(result.is_error)
        self.assertIn("Blocked bash command", tool_result_text(result))
        self.assertEqual(calls, [])

    def test_container_bash_tool_returns_bash_observation(self) -> None:
        from evals.swebench.pro_repo_session import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )

        def run(command, **kwargs):
            del command, kwargs
            return SimpleNamespace(returncode=0, stdout="/app\n", stderr="")

        current = CurrentContainer(name="sal.instance", workdir="/app")
        tool = make_container_bash_tool(current, DockerCommandRunner(run=run))

        result = tool.execute("call", {"command": "pwd"}, lambda: False, None)

        self.assertFalse(result.is_error)
        self.assertIn("$ pwd", text_of(result.content))
        self.assertIn("/app", text_of(result.content))

    def test_container_bash_tool_attach_inlines_image_like_standard_tool(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )

        encoded_png = base64.b64encode(b"png-bytes").decode("ascii")

        def run(command, **kwargs):
            del kwargs
            shell = command[-1]
            if "base64" in shell:
                return SimpleNamespace(returncode=0, stdout=encoded_png, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        current = CurrentContainer(name="sal.instance", workdir="/app")
        tool = make_container_bash_tool(current, DockerCommandRunner(run=run))

        result = tool.execute(
            "call",
            {"command": "true", "description": "noop", "attach": ["plot.png"]},
            lambda: False,
            None,
        )

        image_blocks = [block for block in result.content if block.kind == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0].mime_type, "image/png")
        self.assertEqual(image_blocks[0].data, encoded_png)


def kwargs_command(command: list[str]) -> list[str]:
    """Tiny helper so the test asserts the command value is a plain list."""

    return command


class RepoSessionStateTest(unittest.TestCase):
    def test_instance_tasks_are_appended_to_one_persistent_state(self) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )

        observed_visible_counts: list[int] = []

        def brain(visible):
            observed_visible_counts.append(len(visible))
            return message("done")

        def message(text: str):
            from simple_agent_lab.messages import assistant_message

            return assistant_message(
                text, sender="swebench_agent", target="user", kind="final"
            )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        agent = Agent("swebench_agent", brain)

        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="one",
            task="first problem",
        )
        list(run(agent, state, max_turns=1))

        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="two",
            task="second problem",
        )
        list(run(agent, state, max_turns=1))

        visible_text = "\n".join(message_text(m) for m in state.messages)
        self.assertNotIn("SWE-bench Pro repo session for acme/widgets", visible_text)
        self.assertIn("first problem", visible_text)
        self.assertIn("second problem", visible_text)
        self.assertGreater(observed_visible_counts[1], observed_visible_counts[0])
        instance_messages = [
            message
            for message in state.messages
            if getattr(message, "sidecar", {})
            .get("details", {})
            .get("swebench", {})
            .get("instance_id")
            in {"one", "two"}
        ]
        self.assertEqual(
            [message.kind for message in instance_messages], ["task", "task"]
        )

    def test_repo_session_state_starts_without_extra_model_visible_prompt(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import start_repo_state

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")

        self.assertEqual(state.active_context_messages(), [])
        self.assertIn("SWE-bench Pro repo session for acme/widgets", state.task)


class RepoSessionTrajectoryOutputTest(unittest.TestCase):
    def test_trajectory_export_is_opt_in_for_repo_session_runs(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import _maybe_write_trace

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"

            written = _maybe_write_trace(
                argparse.Namespace(write_trajectories=False),
                path=path,
                state=SimpleNamespace(task="task", events=[], messages=[]),
                trace_id="trace",
                meta={},
            )

            self.assertFalse(written)
            self.assertFalse(path.exists())

    def test_trajectory_export_can_be_enabled_for_debugging(self) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import _maybe_write_trace

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"

            written = _maybe_write_trace(
                argparse.Namespace(write_trajectories=True),
                path=path,
                state=SimpleNamespace(task="task", events=[], messages=[]),
                trace_id="trace",
                meta={"repo": "acme/widgets"},
            )

            self.assertTrue(written)
            self.assertTrue(path.exists())
            self.assertIn('"trace_id": "trace"', path.read_text(encoding="utf-8"))


class RepoSessionInvalidPromptHandlingTest(unittest.TestCase):
    def test_invalid_prompt_error_detection_matches_provider_error_shapes(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _is_invalid_prompt_error,
        )

        self.assertTrue(_is_invalid_prompt_error(RuntimeError("invalid_prompt")))
        self.assertTrue(_is_invalid_prompt_error(RuntimeError("code=-4321")))
        self.assertFalse(_is_invalid_prompt_error(RuntimeError("rate limit")))

    def test_remaining_turn_budget_counts_failed_invalid_prompt_attempts(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import _remaining_turn_budget

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        state.record_event(TurnStartEvent(agent="swebench_agent"))
        event_start = len(state.events)
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )

        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 3), 3)

        state.record_event(TurnStartEvent(agent="swebench_agent"))
        state.record_event(TurnStartEvent(agent="swebench_agent"))

        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 3), 1)
        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 2), 0)

    def test_invalid_prompt_source_is_instance_task_for_current_problem(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import _invalid_prompt_source

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )

        self.assertEqual(
            _invalid_prompt_source(state, instance_id="case-1"),
            "instance_task",
        )

    def test_invalid_prompt_source_is_tool_output_for_latest_tool_result(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import _invalid_prompt_source

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )
        state.record(
            assistant_message(
                (ToolCallBlock(id="call-1", name="bash", arguments={}),),
                sender="swebench_agent",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="call-1",
                        tool_name="bash",
                        content=(TextBlock("bad provider-triggering output"),),
                    )
                ],
                target="swebench_agent",
            )
        )

        self.assertEqual(
            _invalid_prompt_source(state, instance_id="case-1"),
            "tool_output",
        )

    def test_replace_latest_tool_exchange_removes_call_and_result_with_reminder(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import (
            INVALID_PROMPT_TOOL_REMINDER,
            _replace_latest_tool_exchange_for_invalid_prompt,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )
        state.record(
            assistant_message(
                (ToolCallBlock(id="call-1", name="bash", arguments={}),),
                sender="swebench_agent",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="call-1",
                        tool_name="bash",
                        content=(TextBlock("bad provider-triggering output"),),
                    )
                ],
                target="swebench_agent",
            )
        )

        self.assertTrue(
            _replace_latest_tool_exchange_for_invalid_prompt(
                state, agent_name="swebench_agent"
            )
        )

        self.assertFalse(
            any(
                is_tool_result_message(message)
                for message in state.active_context_messages()
            )
        )
        self.assertFalse(
            any(
                message_tool_calls(message)
                for message in state.active_context_messages()
            )
        )
        visible = "\n".join(message_text(m) for m in state.active_context_messages())
        self.assertNotIn("bad provider-triggering output", visible)
        self.assertIn(INVALID_PROMPT_TOOL_REMINDER, visible)
        self.assertTrue(
            any(
                isinstance(event, ContextCompressionEvent)
                and event.strategy == "invalid-prompt-tool-exchange-replace"
                for event in state.events
            )
        )

    def test_repair_active_tool_pairs_drops_orphan_tool_call_before_request(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _repair_active_tool_pairs,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )
        state.record(
            assistant_message(
                (ToolCallBlock(id="orphan-call", name="bash", arguments={}),),
                sender="swebench_agent",
                target="user",
                kind="step",
            )
        )
        state.record(user_message("continue", target="swebench_agent"))

        self.assertTrue(_repair_active_tool_pairs(state, agent_name="swebench_agent"))

        self.assertFalse(
            any(
                message_tool_calls(message)
                for message in state.active_context_messages()
            )
        )
        visible = "\n".join(message_text(m) for m in state.active_context_messages())
        self.assertIn("Removed an incomplete tool call/tool result exchange", visible)
        self.assertTrue(
            any(
                isinstance(event, ContextCompressionEvent)
                and event.strategy == "tool-pair-orphan-repair"
                for event in state.events
            )
        )

    def test_rewrite_instance_task_after_skip_removes_bad_problem_from_context(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _drop_instance_task_for_invalid_prompt_skip,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="bad provider-triggering problem statement",
        )

        self.assertTrue(
            _drop_instance_task_for_invalid_prompt_skip(
                state,
                agent_name="swebench_agent",
                instance_id="case-1",
            )
        )

        visible = "\n".join(message_text(m) for m in state.active_context_messages())
        self.assertNotIn("bad provider-triggering problem statement", visible)
        self.assertFalse(
            any(
                getattr(message, "role", "") == "user"
                and getattr(message, "sidecar", {})
                .get("details", {})
                .get("swebench", {})
                .get("instance_id")
                == "case-1"
                for message in state.active_context_messages()
            )
        )
        self.assertTrue(
            any(
                isinstance(event, ContextCompressionEvent)
                and event.strategy == "invalid-prompt-instance-task-drop"
                for event in state.events
            )
        )

    def test_end_instance_after_tool_retry_limit_drops_latest_tool_result(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _end_instance_after_invalid_prompt_tool_retry_limit,
            _replace_latest_tool_exchange_for_invalid_prompt,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this SWE-bench instance.",
        )
        state.record(
            assistant_message(
                (ToolCallBlock(id="call-1", name="bash", arguments={}),),
                sender="swebench_agent",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="call-1",
                        tool_name="bash",
                        content=(TextBlock("bad provider-triggering output"),),
                    )
                ],
                target="swebench_agent",
            )
        )
        self.assertTrue(
            _replace_latest_tool_exchange_for_invalid_prompt(
                state, agent_name="swebench_agent"
            )
        )

        self.assertTrue(
            _end_instance_after_invalid_prompt_tool_retry_limit(
                state,
                agent_name="swebench_agent",
                instance_id="case-1",
            )
        )

        self.assertEqual(state.active_context_messages(), [])
        self.assertTrue(
            any(
                isinstance(event, ContextCompressionEvent)
                and event.strategy == "invalid-prompt-clear-context"
                for event in state.events
            )
        )

        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-2",
            task="fresh problem",
        )
        visible = "\n".join(message_text(m) for m in state.active_context_messages())
        self.assertEqual(visible, "fresh problem")


class RepoSessionIncrementalPredictionsTest(unittest.TestCase):
    def test_incremental_predictions_file_is_refreshed_after_each_result(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_sessions import (
            _write_incremental_predictions,
        )
        from simple_agent_lab.trace import read_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "run-1"
            instance_dir = root / run_id / "instance_acme__widgets-1"
            (instance_dir / "input").mkdir(parents=True)
            (instance_dir / "out").mkdir()
            (instance_dir / "input" / "instance.json").write_text(
                '{"instance_id": "instance_acme__widgets-1"}\n',
                encoding="utf-8",
            )
            (instance_dir / "out" / "result.json").write_text(
                '{"model_patch": "diff --git a/a b/a\\n"}\n',
                encoding="utf-8",
            )
            predictions_path = root / run_id / "run-1_predictions.jsonl"

            _write_incremental_predictions(
                predictions_path=predictions_path,
                run_root=root,
                run_id=run_id,
                model_name="model",
                dataset_name="ScaleAI/SWE-bench_Pro",
                lock=None,
            )

            predictions = read_jsonl(predictions_path)
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["instance_id"], "instance_acme__widgets-1")


class CommitTimeResolverTest(unittest.TestCase):
    def test_commit_time_resolver_shallow_fetches_commits_in_one_batch(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import CommitTimeResolver

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            del kwargs
            if "cat-file" in command:
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            if "show" in command:
                timestamps = {"abc123": "123\n", "def456": "456\n"}
                return SimpleNamespace(
                    returncode=0, stdout=timestamps[command[-1]], stderr=""
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            resolver = CommitTimeResolver(cache_root=Path(tmp), run=run)
            timestamps = resolver.timestamps(
                "acme/widgets", ["abc123", "def456", "abc123"]
            )

        repo_dir = str(Path(tmp) / "acme__widgets")
        self.assertEqual(timestamps, {"abc123": 123, "def456": 456})
        self.assertEqual(
            calls[0],
            ["git", "-C", repo_dir, "init", "-q"],
        )
        self.assertIn(
            [
                "git",
                "-C",
                repo_dir,
                "remote",
                "add",
                "origin",
                "https://github.com/acme/widgets.git",
            ],
            calls,
        )
        fetch_calls = [command for command in calls if "fetch" in command]
        self.assertEqual(
            fetch_calls,
            [
                [
                    "git",
                    "-C",
                    repo_dir,
                    "-c",
                    "protocol.version=2",
                    "fetch",
                    "--filter=blob:none",
                    "--no-tags",
                    "--depth=1",
                    "origin",
                    "abc123",
                    "def456",
                ]
            ],
        )

    def test_commit_time_resolver_reads_cached_commit_without_fetching(self) -> None:
        from evals.swebench.pro_repo_session import CommitTimeResolver

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            del kwargs
            if "cat-file" in command:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "show" in command:
                return SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "acme__widgets"
            (repo_dir / ".git").mkdir(parents=True)
            resolver = CommitTimeResolver(cache_root=Path(tmp), run=run)
            timestamp = resolver.timestamp("acme/widgets", "abc123")

        self.assertEqual(timestamp, 123)
        self.assertFalse(any("fetch" in command for command in calls))

    def test_commit_time_resolver_returns_none_and_records_warning_on_failure(
        self,
    ) -> None:
        from evals.swebench.pro_repo_session import CommitTimeResolver

        def run(command, **kwargs):
            del command, kwargs
            return SimpleNamespace(returncode=2, stdout="", stderr="boom")

        with tempfile.TemporaryDirectory() as tmp:
            resolver = CommitTimeResolver(cache_root=Path(tmp), run=run)
            timestamp = resolver.timestamp("acme/widgets", "abc123")

        self.assertIsNone(timestamp)
        self.assertTrue(resolver.warnings)
        self.assertIn("acme/widgets", resolver.warnings[0])


if __name__ == "__main__":
    unittest.main()
