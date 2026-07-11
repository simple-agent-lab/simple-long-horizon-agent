from __future__ import annotations

import argparse
import base64
import contextlib
import io
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

SWEBENCH_CONTAINER = "simple_agent_lab.evals.suites.swebench.container"


class SwebenchProRepoChainPlanningTest(unittest.TestCase):
    def test_groups_by_repo_and_sorts_by_commit_timestamp(self) -> None:
        from evals.swebench.pro_repo_chain import (
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
        from evals.swebench.pro_repo_chain import sort_repo_instances

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

    def test_repo_chain_part_count_uses_long_tail_thresholds(self) -> None:
        from evals.swebench.pro_repo_chain import repo_chain_part_count

        self.assertEqual(repo_chain_part_count(20), 1)
        self.assertEqual(repo_chain_part_count(50), 1)
        self.assertEqual(repo_chain_part_count(51), 2)
        self.assertEqual(repo_chain_part_count(80), 2)
        self.assertEqual(repo_chain_part_count(81), 3)

    def test_split_repo_chain_parts_preserves_order_in_near_even_chunks(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import split_repo_chain_parts

        rows = [
            {"instance_id": f"case-{index:03d}", "repo": "acme/widgets"}
            for index in range(85)
        ]

        parts = split_repo_chain_parts("acme/widgets", rows)

        self.assertEqual([part.part_index for part in parts], [1, 2, 3])
        self.assertEqual([part.part_count for part in parts], [3, 3, 3])
        self.assertEqual([len(part.rows) for part in parts], [29, 28, 28])
        self.assertEqual(parts[0].rows[0]["instance_id"], "case-000")
        self.assertEqual(parts[1].rows[0]["instance_id"], "case-029")
        self.assertEqual(parts[2].rows[-1]["instance_id"], "case-084")

    def test_current_pro_dataset_counts_resolve_to_23_chain_parts(self) -> None:
        from evals.swebench.pro_repo_chain import repo_chain_part_count

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
            sum(repo_chain_part_count(n) for n in repo_counts.values()), 23
        )

    def test_default_experiment_config_matches_requested_compression_setup(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            DEFAULT_MODEL_NAME,
            ProRepoExperimentConfig,
        )

        config = ProRepoExperimentConfig()

        self.assertEqual(config.model, "")
        self.assertEqual(config.api_kind, "openai-responses")
        self.assertEqual(config.reasoning_effort, "")
        self.assertEqual(config.threshold_tokens, 217_600)
        self.assertEqual(config.keep_recent, 4)
        self.assertEqual(config.max_turns, 250)
        self.assertEqual(config.agent_flavor, "bash")
        self.assertEqual(config.solver_read, False)
        self.assertEqual(config.task_tool, False)
        self.assertEqual(config.compression_strategy, "none")
        self.assertEqual(config.handoff, True)
        # Handoff trigger defaults to the summarize threshold (272k * 0.8) so the
        # two context-management arms fire at the same point.
        self.assertEqual(config.context_window_tokens, 217_600)
        self.assertEqual(config.context_window_tokens, config.threshold_tokens)
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.preserve_kinds, ("task", "system", "context"))

    def test_build_summarize_policy_uses_same_provider_and_records_parameters(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import ProRepoExperimentConfig

        provider = Provider(
            id="openai-chat",
            api="openai-chat",
            model="gpt-5.4-2026-03-05",
            api_key_env="OPENAI_AUTH_TOKEN",
            default_reasoning="high",
        )
        config = ProRepoExperimentConfig(compression_strategy="summarize")

        policy = config.context_policy(provider, request_extra={"x": "y"})

        strategy = policy.strategy
        self.assertIsInstance(strategy, SummarizeStrategy)
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 217_600)
        self.assertEqual(strategy.keep_recent, 4)
        self.assertEqual(strategy.preserve_kinds, ("task", "system", "context"))
        self.assertEqual(strategy.compressor.name, "swebench_compressor")
        self.assertEqual(strategy.compressor.context_policy, None)

    def test_repo_chain_runner_defaults_to_openai_responses_api_kind(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertEqual(args.api_kind, "openai-responses")
        self.assertEqual(args.parallel, "slots")
        self.assertIsNone(args.chains_json)

    def test_repo_chain_runner_requires_explicit_chains_json_to_load(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _load_chains,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])

        with self.assertRaises(SystemExit) as raised:
            _load_chains(args)

        self.assertIn("Pass --chains-json PATH", str(raised.exception))

    def test_repo_chain_runner_leaves_model_and_reasoning_to_env_by_default(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertIsNone(args.model)
        self.assertIsNone(args.reasoning_effort)
        self.assertIsNone(args.provider_auth_envs)

    def test_repo_chain_runner_accepts_task_tool_as_a_variable(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--all",
                "--task-tool",
            ]
        )
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.agent_flavor, "bash")
        self.assertTrue(config.task_tool)

    def test_repo_chain_runner_accepts_non_goal_agent_flavor(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all", "--agent-flavor", "bash"])
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.agent_flavor, "bash")
        self.assertEqual(config.solver_read, False)

    def test_repo_chain_runner_rejects_read_tool_flavors(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import build_parser

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            build_parser().parse_args(["--all", "--agent-flavor", "bash_task_read"])

    def test_default_run_id_is_derived_from_selected_variables(self) -> None:
        from datetime import datetime

        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            _resolve_run_id,
            build_parser,
        )

        now = datetime(2026, 6, 30, 12, 34, 56)
        compression_args = build_parser().parse_args(
            ["--all", "--compression-strategy", "summarize"]
        )
        compression_config = _experiment_config_from_args(
            compression_args, api_kind="openai-responses"
        )
        chain_args = build_parser().parse_args(
            [
                "--all",
                "--compression-strategy",
                "none",
                "--task-tool",
            ]
        )
        chain_config = _experiment_config_from_args(
            chain_args, api_kind="openai-responses"
        )
        goal_args = build_parser().parse_args(
            [
                "--all",
                "--agent-flavor",
                "goal",
                "--compression-strategy",
                "none",
                "--task-tool",
            ]
        )
        goal_config = _experiment_config_from_args(
            goal_args, api_kind="openai-responses"
        )

        self.assertEqual(
            _resolve_run_id(compression_args.run_id, compression_config, now=now),
            "pro-repo-chain-summarize-20260630-123456",
        )
        self.assertEqual(
            _resolve_run_id(chain_args.run_id, chain_config, now=now),
            "pro-repo-chain-task-none-20260630-123456",
        )
        self.assertEqual(
            _resolve_run_id(goal_args.run_id, goal_config, now=now),
            "pro-repo-chain-goal-task-none-20260630-123456",
        )
        self.assertEqual(
            _resolve_run_id("manual-run", chain_config, now=now), "manual-run"
        )

    def test_model_name_records_chain_agent_tools_and_compression(self) -> None:
        from evals.swebench.pro_repo_chain import DEFAULT_MODEL_NAME
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        default_args = build_parser().parse_args(["--all"])
        bash_args = build_parser().parse_args(
            [
                "--all",
                "--agent-flavor",
                "bash",
                "--task-tool",
                "--compression-strategy",
                "none",
            ]
        )

        default_config = _experiment_config_from_args(
            default_args, api_kind="openai-responses"
        )
        bash_config = _experiment_config_from_args(
            bash_args, api_kind="openai-responses"
        )

        self.assertEqual(
            DEFAULT_MODEL_NAME,
            "simple-agent-lab-pro-repo-chain-bash-none",
        )
        self.assertEqual(default_config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(
            bash_config.model_name,
            "simple-agent-lab-pro-repo-chain-bash-task-none",
        )

    def test_provider_auth_slots_expand_counts(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _expand_auth_slots,
        )

        auth_envs = _expand_auth_slots("OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11")

        self.assertEqual(auth_envs[:12], ["OPENAI_AUTH_TOKEN"] * 12)
        self.assertEqual(auth_envs[12:], ["OPENAI_AUTH_TOKEN2"] * 11)

    def test_provider_auth_slots_use_primary_token_by_default(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _expand_auth_slots,
        )

        self.assertEqual(_expand_auth_slots(None), ["OPENAI_AUTH_TOKEN"])

    def test_provider_slots_use_api_kind_temperature_defaults(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
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

    def test_provider_auth_slots_reject_bad_specs(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _expand_auth_slots,
        )

        with self.assertRaises(SystemExit):
            _expand_auth_slots("OPENAI_AUTH_TOKEN:0")

    def test_parallel_slots_use_provider_lane_count_for_memory_order(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _expand_auth_slots,
            _resolve_parallel,
        )

        slots = _expand_auth_slots("OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11")

        self.assertEqual(len(slots), 23)
        self.assertEqual(
            _resolve_parallel("slots", chain_count=100, slot_count=len(slots)),
            23,
        )
        self.assertEqual(
            _resolve_parallel("parts", chain_count=5, slot_count=len(slots)),
            5,
        )
        self.assertEqual(_resolve_parallel("8", chain_count=100, slot_count=23), 8)

    def test_experiment_config_records_effective_env_model_and_reasoning(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
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

    def test_repo_chain_runner_allows_prior_summaries_to_be_resummarized(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])

        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.preserve_kinds, ("task", "system", "context"))

    def test_responses_request_extra_includes_encrypted_reasoning(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _request_extra_for_api_kind,
        )

        extra = _request_extra_for_api_kind("openai-responses")

        self.assertEqual(extra["include"], ["reasoning.encrypted_content"])

    def test_non_responses_request_extra_does_not_include_encrypted_reasoning(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _request_extra_for_api_kind,
        )

        extra = _request_extra_for_api_kind("openai-chat")

        self.assertNotIn("include", extra)

    def test_cli_model_and_reasoning_override_env_when_explicit(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
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

    def test_goal_flavor_maps_max_turns_to_worker_and_defaults_loop_to_one(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _apply_provider_env_overrides,
            build_parser,
        )

        args = build_parser().parse_args(
            ["--all", "--agent-flavor", "goal", "--max-turns", "250"]
        )

        with patch.dict(os.environ, {}, clear=True):
            _apply_provider_env_overrides(args)

            self.assertEqual(os.environ["SAL_WORKFLOW_WORKER_MAX_TURNS"], "250")
            self.assertEqual(os.environ["SAL_WORKFLOW_LOOP_MAX_TURNS"], "1")

    def test_goal_max_turns_overrides_env_worker_but_loop_env_wins(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _apply_provider_env_overrides,
            build_parser,
        )

        args = build_parser().parse_args(
            ["--all", "--agent-flavor", "goal", "--max-turns", "120"]
        )

        with patch.dict(
            os.environ,
            {
                "SAL_WORKFLOW_WORKER_MAX_TURNS": "40",
                "SAL_WORKFLOW_LOOP_MAX_TURNS": "6",
            },
            clear=True,
        ):
            _apply_provider_env_overrides(args)

            self.assertEqual(os.environ["SAL_WORKFLOW_WORKER_MAX_TURNS"], "120")
            self.assertEqual(os.environ["SAL_WORKFLOW_LOOP_MAX_TURNS"], "6")

    def test_non_goal_flavor_leaves_goal_turn_budget_untouched(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _apply_provider_env_overrides,
            build_parser,
        )

        args = build_parser().parse_args(
            ["--all", "--agent-flavor", "bash", "--max-turns", "250"]
        )

        with patch.dict(os.environ, {}, clear=True):
            _apply_provider_env_overrides(args)

            self.assertNotIn("SAL_WORKFLOW_WORKER_MAX_TURNS", os.environ)
            self.assertNotIn("SAL_WORKFLOW_LOOP_MAX_TURNS", os.environ)

    def test_selected_provider_auth_slot_is_mapped_to_container_primary_auth(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _provider_env_for_auth_env,
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "model",
                "OPENAI_AUTH_TOKEN2": "token-2",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
                "REASONING_EFFORT": "high",
            },
            clear=True,
        ):
            env = _provider_env_for_auth_env(
                "OPENAI_AUTH_TOKEN2",
                api_kind="openai-responses",
            )

        self.assertEqual(env["OPENAI_AUTH_TOKEN"], "token-2")
        self.assertEqual(env["OPENAI_MODEL"], "model")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://example.invalid/v1")
        self.assertEqual(env["REASONING_EFFORT"], "high")
        self.assertEqual(env["API_KIND"], "openai-responses")

    def test_chain_config_payload_is_staged_for_container_runner(self) -> None:
        from evals.swebench.pro_repo_chain import (
            ProRepoExperimentConfig,
            RepoChainPart,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _chain_config_payload,
        )

        chain = RepoChainPart(
            chain_id="acme/widgets#part-1-of-2",
            repo="acme/widgets",
            part_index=1,
            part_count=2,
            rows=({"instance_id": "case-1"},),
        )
        config = ProRepoExperimentConfig(threshold_tokens=123, keep_recent=7)

        payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env="OPENAI_AUTH_TOKEN2",
            position=1,
            write_trajectories=False,
        )

        self.assertEqual(payload["mode"], "repo_chain")
        self.assertEqual(payload["repo"], "acme/widgets")
        self.assertEqual(payload["chain_id"], "acme/widgets#part-1-of-2")
        self.assertEqual(payload["chain_display_name"], "acme/widgets part 1/2")
        self.assertEqual(payload["part_index"], 1)
        self.assertEqual(payload["part_count"], 2)
        self.assertEqual(payload["provider_auth_env"], "OPENAI_AUTH_TOKEN2")
        self.assertEqual(payload["config"]["agent_flavor"], "bash")
        self.assertEqual(payload["config"]["solver_read"], False)
        self.assertEqual(payload["config"]["task_tool"], False)
        self.assertEqual(payload["config"]["threshold_tokens"], 123)
        self.assertEqual(payload["config"]["keep_recent"], 7)

    def test_chain_config_payload_records_selected_agent_flavor(self) -> None:
        from evals.swebench.pro_repo_chain import (
            ProRepoExperimentConfig,
            RepoChainPart,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _chain_config_payload,
        )

        chain = RepoChainPart(
            chain_id="acme/widgets",
            repo="acme/widgets",
            part_index=1,
            part_count=1,
            rows=({"instance_id": "case-1"},),
        )
        config = ProRepoExperimentConfig(agent_flavor="bash")

        payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env="OPENAI_AUTH_TOKEN",
            position=1,
            write_trajectories=False,
        )

        self.assertEqual(payload["config"]["agent_flavor"], "bash")
        self.assertEqual(payload["config"]["solver_read"], False)
        self.assertEqual(payload["config"]["task_tool"], False)

    def test_chain_config_payload_records_bash_only_defaults(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            ProRepoExperimentConfig,
            RepoChainPart,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _chain_config_payload,
        )

        chain = RepoChainPart(
            chain_id="acme/widgets",
            repo="acme/widgets",
            part_index=1,
            part_count=1,
            rows=({"instance_id": "case-1"},),
        )
        config = ProRepoExperimentConfig()

        payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env="OPENAI_AUTH_TOKEN",
            position=1,
            write_trajectories=False,
        )

        self.assertEqual(payload["mode"], "repo_chain")
        self.assertEqual(payload["config"]["agent_flavor"], "bash")
        self.assertEqual(payload["config"]["solver_read"], False)
        self.assertEqual(payload["config"]["task_tool"], False)

    def test_chain_config_payload_records_task_tool_variable(self) -> None:
        from evals.swebench.pro_repo_chain import (
            ProRepoExperimentConfig,
            RepoChainPart,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _chain_config_payload,
        )

        chain = RepoChainPart(
            chain_id="acme/widgets",
            repo="acme/widgets",
            part_index=1,
            part_count=1,
            rows=({"instance_id": "case-1"},),
        )
        config = ProRepoExperimentConfig(
            compression_strategy="none",
            task_tool=True,
        )

        payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env="OPENAI_AUTH_TOKEN",
            position=1,
            write_trajectories=False,
        )

        self.assertEqual(payload["config"]["task_tool"], True)

    def test_write_json_replaces_existing_unwritable_artifact(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import _write_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "result.json"
            path.parent.mkdir()
            path.write_text('{"old": true}\n', encoding="utf-8")
            path.chmod(0o444)
            try:
                _write_json(path, {"new": True})
            finally:
                path.chmod(0o644)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "new": true\n}\n',
            )

    def test_plan_manifest_records_selected_agent_and_compression_mode(self) -> None:
        from evals.swebench.pro_memory_chain import DEFAULT_CHAINS_JSON, RawIssueChain
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            _plan_groups,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--all",
                "--chains-json",
                str(DEFAULT_CHAINS_JSON),
                "--compression-strategy",
                "none",
                "--task-tool",
            ]
        )
        config = _experiment_config_from_args(args, api_kind="openai-responses")
        rows = [
            {"instance_id": "a", "repo": "acme/widgets", "base_commit": ""},
            {"instance_id": "b", "repo": "acme/widgets", "base_commit": ""},
            {"instance_id": "c", "repo": "acme/widgets", "base_commit": ""},
            {"instance_id": "solo", "repo": "acme/widgets", "base_commit": ""},
        ]
        raw_chains = [
            RawIssueChain(
                chain_id="long",
                repo="acme/widgets",
                instance_ids=("a", "b", "c"),
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            chains, manifest = _plan_groups(
                rows,
                raw_chains=raw_chains,
                args=args,
                run_root=Path(tmp),
                config=config,
            )

        self.assertEqual(list(chains), ["long", "solo"])
        self.assertEqual(manifest["config"]["agent_flavor"], "bash")
        self.assertEqual(manifest["config"]["compression_strategy"], "none")
        self.assertEqual(manifest["config"]["task_tool"], True)
        self.assertEqual(manifest["plan_source"], "memory_issue_chains")
        self.assertEqual(manifest["run_unit_count"], 2)
        self.assertEqual(manifest["chain_count"], 1)
        self.assertEqual(manifest["memory_chain_count"], 1)
        self.assertEqual(manifest["singleton_count"], 1)
        self.assertEqual(manifest["chain_length_histogram"], {"3": 1})
        self.assertEqual(
            manifest["per_repo"],
            {"acme/widgets": {"chains": 1, "chain_instances": 3, "singletons": 1}},
        )
        self.assertEqual(
            [entry["instance_ids"] for entry in manifest["order"]],
            [["a", "b", "c"], ["solo"]],
        )
        self.assertEqual(
            [entry["length"] for entry in manifest["order"]],
            [3, 1],
        )
        self.assertEqual(
            [entry["memory_enabled"] for entry in manifest["order"]],
            [False, False],
        )

    def test_select_units_applies_max_chains_and_limit(self) -> None:
        from evals.swebench.pro_memory_chain import (
            RawIssueChain,
            plan_memory_chains,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _select_units,
            build_parser,
        )

        rows = [
            {"instance_id": "a1", "repo": "r", "base_commit": "1"},
            {"instance_id": "a2", "repo": "r", "base_commit": "2"},
            {"instance_id": "a3", "repo": "r", "base_commit": "3"},
            {"instance_id": "b1", "repo": "r", "base_commit": "4"},
            {"instance_id": "b2", "repo": "r", "base_commit": "5"},
            {"instance_id": "solo", "repo": "r", "base_commit": "6"},
        ]
        raw_chains = [
            RawIssueChain(chain_id="c-a", repo="r", instance_ids=("a1", "a2", "a3")),
            RawIssueChain(chain_id="c-b", repo="r", instance_ids=("b1", "b2")),
        ]
        plan = plan_memory_chains(rows, raw_chains, memory=False)

        max_chain_args = build_parser().parse_args(["--all", "--max-chains", "1"])
        kept = _select_units(plan.chains, args=max_chain_args)
        self.assertEqual(
            [unit.chain_id for unit in kept if not unit.is_singleton], ["c-a"]
        )
        self.assertTrue(any(unit.is_singleton for unit in kept))

        limit_args = build_parser().parse_args(["--all", "--limit", "2"])
        limited = _select_units(plan.chains, args=limit_args)
        self.assertEqual([unit.instance_ids for unit in limited], [["a1", "a2"]])

    def test_handoff_is_enabled_by_default_with_default_window(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertTrue(args.handoff)
        self.assertEqual(args.context_window_tokens, 217_600)
        self.assertTrue(config.handoff)
        self.assertEqual(config.context_window_tokens, 217_600)
        self.assertEqual(config.as_record()["handoff"], True)
        self.assertEqual(config.as_record()["context_window_tokens"], 217_600)

    def test_no_handoff_flag_disables_handoff(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all", "--no-handoff"])
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertFalse(args.handoff)
        self.assertFalse(config.handoff)
        self.assertEqual(config.as_record()["handoff"], False)

    def test_context_window_tokens_flag_flows_into_config(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all", "--context-window-tokens", "5000"])
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(args.context_window_tokens, 5000)
        self.assertEqual(config.context_window_tokens, 5000)
        self.assertEqual(config.as_record()["context_window_tokens"], 5000)

    def test_chain_config_payload_carries_handoff_and_window_tokens(self) -> None:
        from evals.swebench.pro_repo_chain import (
            ProRepoExperimentConfig,
            RepoChainPart,
        )
        from runs.swebench.run_swebench_pro_repo_chains import (
            _chain_config_payload,
        )

        chain = RepoChainPart(
            chain_id="acme/widgets",
            repo="acme/widgets",
            part_index=1,
            part_count=1,
            rows=({"instance_id": "case-1"},),
        )
        config = ProRepoExperimentConfig(handoff=False, context_window_tokens=1234)

        payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env="OPENAI_AUTH_TOKEN",
            position=1,
            write_trajectories=False,
        )

        self.assertEqual(payload["config"]["handoff"], False)
        self.assertEqual(payload["config"]["context_window_tokens"], 1234)


class DockerExecBashToolTest(unittest.TestCase):
    def test_docker_runner_executes_command_with_standard_bash_shell(self) -> None:
        from evals.swebench.pro_repo_chain import DockerCommandRunner

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
        from evals.swebench.pro_repo_chain import (
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

    def test_docker_runner_starts_pro_container_for_long_lived_chain(self) -> None:
        from evals.swebench.pro_repo_chain import DockerCommandRunner

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
                "16g",
                "-w",
                "/app",
                "jefzda/sweap-images:repo.commit",
                "/bin/sh",
                "-lc",
                "sleep infinity",
            ],
        )

    def test_docker_runner_removes_container_forcefully(self) -> None:
        from evals.swebench.pro_repo_chain import DockerCommandRunner

        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        runner = DockerCommandRunner(run=run)

        runner.remove_container("sal.instance")

        self.assertEqual(calls[0], ["docker", "rm", "-f", "sal.instance"])

    def test_prepare_container_baseline_uses_long_timeout_for_large_workspaces(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            CurrentContainer,
            prepare_container_baseline,
        )

        calls: list[dict[str, object]] = []

        class Docker:
            def exec(self, **kwargs):
                calls.append(dict(kwargs))
                if "git rev-parse HEAD" in str(kwargs["command"]):
                    return SimpleNamespace(
                        returncode=0, stdout="baseline-sha\n", stderr=""
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        baseline = prepare_container_baseline(
            Docker(),
            CurrentContainer(name="sal.instance", workdir="/app"),
            language="js",
        )

        self.assertEqual(baseline, "baseline-sha")
        self.assertEqual(calls[0]["timeout_seconds"], 30)
        self.assertGreaterEqual(calls[1]["timeout_seconds"], 300)

    def test_container_bash_tool_reports_missing_current_container(self) -> None:
        from evals.swebench.pro_repo_chain import (
            CurrentContainer,
            DockerCommandRunner,
            make_container_bash_tool,
        )

        tool = make_container_bash_tool(
            CurrentContainer(), DockerCommandRunner(run=lambda *a, **k: None)
        )

        result = tool.execute("call", {"command": "pwd"}, lambda: False, None)

        self.assertTrue(result.is_error)
        self.assertIn("No active repository container", text_of(result.content))

    def test_container_bash_tool_blocks_long_leading_sleep_like_standard_tool(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
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
        from evals.swebench.pro_repo_chain import (
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
        from evals.swebench.pro_repo_chain import (
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


class RepoChainStateTest(unittest.TestCase):
    def test_instance_tasks_are_appended_to_one_persistent_state(self) -> None:
        from evals.swebench.pro_repo_chain import (
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
        self.assertNotIn("Repo chain for acme/widgets", visible_text)
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
            [message.kind for message in instance_messages],
            ["task", "message", "task"],
        )
        active_instance_messages = [
            message
            for message in state.active_context_messages()
            if getattr(message, "sidecar", {})
            .get("details", {})
            .get("swebench", {})
            .get("instance_id")
            in {"one", "two"}
        ]
        self.assertEqual(
            [message.kind for message in active_instance_messages],
            ["message", "task"],
        )

    def test_repo_chain_state_starts_without_extra_model_visible_prompt(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import start_repo_state

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")

        self.assertEqual(state.active_context_messages(), [])
        self.assertIn("Repo chain for acme/widgets", state.task)


class RepoChainStateArtifactTest(unittest.TestCase):
    def test_appending_new_chain_task_demotes_prior_item_tasks(self) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="first problem",
        )
        state.record(
            assistant_message(
                "first answer",
                sender="swebench_agent",
                target="user",
                kind="final",
            )
        )
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-2",
            task="second problem",
        )

        active = state.active_context_messages()

        self.assertEqual(
            [message_text(message) for message in active],
            [
                "first problem",
                "first answer",
                "second problem",
            ],
        )
        self.assertEqual(active[0].kind, "message")
        self.assertEqual(active[2].kind, "task")
        demotions = [
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
            and event.strategy == "chain-task-demote"
        ]
        self.assertEqual(len(demotions), 1)

    def test_chain_payload_round_trips_active_context_and_metadata(self) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            state_from_chain_payload,
            state_to_chain_payload,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        state.data["eval_chain"]["chain_id"] = "acme/widgets#part-1-of-2"
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="first problem",
        )
        state.record(
            assistant_message(
                "first answer",
                sender="swebench_agent",
                target="user",
                kind="final",
            )
        )
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-2",
            task="second problem",
        )
        state.record_event(
            ContextCompressionEvent(
                agent="swebench_agent",
                summary_message_index=3,
                compressed_message_indices=[1, 2],
                active_context_indices=[3],
                before_tokens=100,
                after_tokens=10,
                strategy="test-compact",
            )
        )

        payload = state_to_chain_payload(state)
        restored = state_from_chain_payload(payload)

        self.assertEqual(restored.task, state.task)
        self.assertEqual(
            restored.data["eval_chain"]["chain_id"],
            "acme/widgets#part-1-of-2",
        )
        self.assertEqual(
            [message_text(message) for message in restored.active_context_messages()],
            ["second problem"],
        )
        active = restored.active_context_messages()[0]
        self.assertEqual(active.kind, "task")
        self.assertEqual(
            active.sidecar["details"]["swebench"]["instance_id"],
            "case-2",
        )
        self.assertLess(len(payload["messages"]), len(state.messages))


class RepoChainInContainerRunnerTest(unittest.TestCase):
    def test_goal_chain_delegates_to_run_goal_flavor_on_shared_state(
        self,
    ) -> None:
        import json

        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals import LocalDirStore
        from simple_agent_lab.evals import chain as chain_mod
        from simple_agent_lab.evals.chain import (
            CHAIN_CONFIG_KEY,
            CHAIN_STATE_INPUT_KEY,
            run_chain_in_container,
            state_to_chain_payload,
        )
        from simple_agent_lab.llm.env import FAKE_PROVIDER

        captured: dict[str, object] = {}

        def fake_run_goal_flavor(provider, workdir, request_extra, **kwargs):
            captured.update(kwargs)
            captured["provider"] = provider
            return SimpleNamespace(output="", steps=[])

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                import subprocess

                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            store = LocalDirStore(Path(tmp) / "run")
            prior = start_repo_state("acme/widgets", agent_name="swebench_agent")
            append_instance_task(
                prior,
                agent_name="swebench_agent",
                instance_id="case-0",
                task="prior repo context",
            )
            store.put(
                CHAIN_STATE_INPUT_KEY,
                (
                    json.dumps(state_to_chain_payload(prior), ensure_ascii=False) + "\n"
                ).encode("utf-8"),
            )
            store.put(
                CHAIN_CONFIG_KEY,
                json.dumps(
                    {
                        "mode": "repo_chain",
                        "repo": "acme/widgets",
                        "chain_id": "acme/widgets",
                        "part_index": 1,
                        "part_count": 1,
                        "position": 1,
                        "instances_in_chain": 1,
                        "provider_auth_env": "OPENAI_AUTH_TOKEN",
                        "config": {
                            "agent_flavor": "goal",
                            "solver_read": False,
                            "task_tool": True,
                            "compression_strategy": "none",
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            instance = {
                "instance_id": "case-1",
                "repo": "acme/widgets",
                "problem_statement": "Investigate app.py behavior.",
                "language": "python",
            }

            with patch.object(
                chain_mod, "run_goal_flavor", side_effect=fake_run_goal_flavor
            ):
                result, state = run_chain_in_container(
                    instance=instance,
                    container_module=SWEBENCH_CONTAINER,
                    provider=FAKE_PROVIDER,
                    workdir=repo,
                    max_turns=2,
                    store=store,
                    trace_id="trace.case-1",
                    producer="suite:swebench_pro",
                    suite_name="swebench_pro",
                    request_extra={},
                )

        # Approach A: goal delegates to the ORIGINAL loop (run_goal_flavor) seeded
        # with the SHARED chain state, so it inherits earlier instances' context.
        self.assertEqual(result["agent_flavor"], "goal")
        self.assertIs(captured["state"], state)
        self.assertEqual(captured["steering_preface"], chain_mod.CHAIN_GOAL_PREFACE)
        self.assertEqual(captured["solver_read"], False)
        self.assertEqual(captured["solver_task"], True)
        self.assertIn("Investigate app.py behavior.", str(captured["objective"]))

    def test_bash_chain_builds_bash_agent_with_optional_task(self) -> None:
        from simple_agent_lab.evals.chain import _build_agent
        from simple_agent_lab.llm.env import FAKE_PROVIDER

        captured: dict[str, object] = {}

        def fake_build_flavor_agent(**kwargs):
            captured.update(kwargs)
            return Agent("captured", lambda visible: assistant_message("done"))

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "simple_agent_lab.evals.chain.build_flavor_agent",
                side_effect=fake_build_flavor_agent,
            ),
        ):
            _build_agent(
                provider=FAKE_PROVIDER,
                workdir=Path(tmp),
                request_extra={},
                config={
                    "mode": "chain",
                    "config": {
                        "agent_flavor": "bash",
                        "solver_read": False,
                        "task_tool": True,
                        "compression_strategy": "none",
                    },
                },
                container_module=SWEBENCH_CONTAINER,
            )

        self.assertEqual(captured["flavor"], "bash")
        self.assertEqual(captured["solver_read"], False)
        self.assertEqual(captured["solver_task"], True)

    def test_thread_goal_solver_can_disable_read_and_enable_task_tool(self) -> None:
        from simple_agent_lab.agents.flavors import _solver_agent
        from simple_agent_lab.llm.env import FAKE_PROVIDER

        with tempfile.TemporaryDirectory() as tmp:
            agent = _solver_agent(
                FAKE_PROVIDER,
                Path(tmp),
                {},
                name="goal_agent",
                role="",
                system_prompt="",
                context_policy=None,
                read=False,
                task=True,
            )

        tool_names = sorted(tool.name for tool in agent.tools)
        self.assertEqual(tool_names, ["bash", "task"])

    def test_handoff_active_is_default_on_without_compression(self) -> None:
        from simple_agent_lab.evals.chain import _handoff_active

        config = {
            "mode": "repo_chain",
            "config": {
                "compression_strategy": "none",
                "context_window_tokens": 272_000,
            },
        }

        self.assertTrue(_handoff_active(config))

    def test_handoff_active_is_off_when_handoff_disabled(self) -> None:
        from simple_agent_lab.evals.chain import _handoff_active

        config = {
            "mode": "repo_chain",
            "config": {
                "compression_strategy": "none",
                "context_window_tokens": 272_000,
                "handoff": False,
            },
        }

        self.assertFalse(_handoff_active(config))

    def test_handoff_active_is_off_when_compression_summarizes(self) -> None:
        from simple_agent_lab.evals.chain import _handoff_active

        config = {
            "mode": "repo_chain",
            "config": {
                "compression_strategy": "summarize",
                "context_window_tokens": 272_000,
                "handoff": True,
            },
        }

        self.assertFalse(_handoff_active(config))

    def test_handoff_active_is_off_without_a_window_budget(self) -> None:
        from simple_agent_lab.evals.chain import _handoff_active

        config = {
            "mode": "repo_chain",
            "config": {
                "compression_strategy": "none",
                "context_window_tokens": 0,
                "handoff": True,
            },
        }

        self.assertFalse(_handoff_active(config))

    def test_chain_task_tool_invalid_prompt_error_result_triggers_repair_path(
        self,
    ) -> None:
        from simple_agent_lab.evals.chain import (
            _message_has_invalid_prompt_task_error,
        )
        from simple_agent_lab.messages import tool_result_message
        from simple_agent_lab.protocols import MessageEvent

        task_error = MessageEvent(
            message=tool_result_message(
                "BadRequestError: invalid_prompt",
                tool_call_id="call-1",
                tool_name="task",
                target="swebench",
                is_error=True,
            )
        )
        bash_error = MessageEvent(
            message=tool_result_message(
                "BadRequestError: invalid_prompt",
                tool_call_id="call-2",
                tool_name="bash",
                target="swebench",
                is_error=True,
            )
        )

        self.assertTrue(_message_has_invalid_prompt_task_error(task_error))
        self.assertFalse(_message_has_invalid_prompt_task_error(bash_error))

    def test_runner_continues_state_runs_local_agent_and_writes_artifacts(
        self,
    ) -> None:
        import json

        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals import LocalDirStore, RESULT_KEY
        from simple_agent_lab.evals.chain import (
            run_chain_in_container,
        )
        from simple_agent_lab.evals.chain import (
            CHAIN_CONFIG_KEY,
            CHAIN_STATE_INPUT_KEY,
            CHAIN_STATE_OUTPUT_KEY,
            state_from_chain_payload,
            state_to_chain_payload,
        )
        from simple_agent_lab.llm.env import FAKE_PROVIDER

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                import subprocess

                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            run_dir = Path(tmp) / "run"
            store = LocalDirStore(run_dir)
            prior = start_repo_state("acme/widgets", agent_name="swebench_agent")
            append_instance_task(
                prior,
                agent_name="swebench_agent",
                instance_id="case-0",
                task="prior repo context",
            )
            store.put(
                CHAIN_STATE_INPUT_KEY,
                (
                    json.dumps(state_to_chain_payload(prior), ensure_ascii=False) + "\n"
                ).encode("utf-8"),
            )
            store.put(
                CHAIN_CONFIG_KEY,
                json.dumps(
                    {
                        "mode": "compression",
                        "repo": "acme/widgets",
                        "chain_id": "acme/widgets",
                        "part_index": 1,
                        "part_count": 1,
                        "position": 1,
                        "instances_in_chain": 1,
                        "provider_auth_env": "OPENAI_AUTH_TOKEN",
                        "config": {
                            "compression_strategy": "summarize",
                            "threshold_tokens": 999999,
                            "keep_recent": 4,
                            "preserve_kinds": [
                                "task",
                                "system",
                                "context",
                            ],
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            instance = {
                "instance_id": "case-1",
                "repo": "acme/widgets",
                "problem_statement": (
                    "Change app.py. <bash>printf 'x = 2\\n' > app.py</bash>"
                ),
                "language": "python",
            }

            result, state = run_chain_in_container(
                instance=instance,
                container_module=SWEBENCH_CONTAINER,
                provider=FAKE_PROVIDER,
                workdir=repo,
                max_turns=2,
                store=store,
                trace_id="trace.case-1",
                producer="suite:swebench_pro",
                suite_name="swebench_pro",
                request_extra={},
            )

            self.assertEqual(result["status"], "ok")
            self.assertIn("x = 2", result["model_patch"])
            self.assertIn("diff --git a/app.py b/app.py", result["model_patch"])
            self.assertNotIn("model_submitted_patch", result)
            written_result = json.loads(store.get(RESULT_KEY).decode("utf-8"))
            self.assertEqual(written_result["model_patch"], result["model_patch"])
            self.assertNotIn("model_submitted_patch", written_result)
            restored = state_from_chain_payload(
                json.loads(store.get(CHAIN_STATE_OUTPUT_KEY).decode("utf-8"))
            )
            visible = "\n".join(
                message_text(message) for message in restored.active_context_messages()
            )
            self.assertIn("prior repo context", visible)
            self.assertTrue(
                any(
                    getattr(message, "sidecar", {})
                    .get("details", {})
                    .get("swebench", {})
                    .get("instance_id")
                    == "case-1"
                    for message in restored.active_context_messages()
                )
            )
            self.assertIn("case-1", state.data["eval_chain"]["last_item_id"])

    def test_goal_chain_runs_solver_on_shared_chain_state(self) -> None:
        import json

        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals import LocalDirStore
        from simple_agent_lab.evals.chain import (
            CHAIN_CONFIG_KEY,
            CHAIN_STATE_INPUT_KEY,
            CHAIN_STATE_OUTPUT_KEY,
            run_chain_in_container,
            state_from_chain_payload,
            state_to_chain_payload,
        )
        from simple_agent_lab.llm.env import FAKE_PROVIDER
        from simple_agent_lab.messages import tool_results_of
        from simple_agent_lab.workflow import THREAD_GOAL_STORE_DATA_KEY

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                import subprocess

                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            run_dir = Path(tmp) / "run"
            store = LocalDirStore(run_dir)
            prior = start_repo_state("acme/widgets", agent_name="swebench_agent")
            append_instance_task(
                prior,
                agent_name="swebench_agent",
                instance_id="case-0",
                task="prior repo context",
            )
            store.put(
                CHAIN_STATE_INPUT_KEY,
                (
                    json.dumps(state_to_chain_payload(prior), ensure_ascii=False) + "\n"
                ).encode("utf-8"),
            )
            store.put(
                CHAIN_CONFIG_KEY,
                json.dumps(
                    {
                        "mode": "repo_chain",
                        "repo": "acme/widgets",
                        "chain_id": "acme/widgets",
                        "part_index": 1,
                        "part_count": 1,
                        "position": 1,
                        "instances_in_chain": 1,
                        "provider_auth_env": "OPENAI_AUTH_TOKEN",
                        "config": {
                            "agent_flavor": "goal",
                            "solver_read": False,
                            "task_tool": False,
                            "compression_strategy": "none",
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            instance = {
                "instance_id": "case-1",
                "repo": "acme/widgets",
                "problem_statement": "Investigate app.py behavior.",
                "language": "python",
            }

            # Pin the goal budgets small so the loop runs one short segment: this
            # exercises the inheritance wiring, not the budget mechanics.
            with patch.dict(
                os.environ,
                {
                    "SAL_WORKFLOW_LOOP_MAX_TURNS": "1",
                    "SAL_WORKFLOW_WORKER_MAX_TURNS": "2",
                },
            ):
                result, state = run_chain_in_container(
                    instance=instance,
                    container_module=SWEBENCH_CONTAINER,
                    provider=FAKE_PROVIDER,
                    workdir=repo,
                    max_turns=2,
                    store=store,
                    trace_id="trace.case-1",
                    producer="suite:swebench_pro",
                    suite_name="swebench_pro",
                    request_extra={},
                )

            self.assertEqual(result["agent_flavor"], "goal")
            # The goal store lives on the shared chain state.
            self.assertIn(THREAD_GOAL_STORE_DATA_KEY, state.data)
            restored = state_from_chain_payload(
                json.loads(store.get(CHAIN_STATE_OUTPUT_KEY).decode("utf-8"))
            )
            visible = "\n".join(
                message_text(message) for message in restored.active_context_messages()
            )
            # Earlier instance context is inherited into the current instance.
            self.assertIn("prior repo context", visible)
            # The goal steering framed the run as one long chained problem.
            self.assertIn("chained problem", visible)
            # The bash solver actually ran on the shared chain state, so its tool
            # turns are recorded on the chain itself (the point of goal inheriting
            # the long-chain context instead of a throwaway per-instance state).
            has_bash = any(
                block.tool_name == "bash"
                for message in state.messages
                for block in tool_results_of(message.content)
            )
            self.assertTrue(has_bash)

    def _run_bash_instance_with_handoff_config(
        self,
        *,
        context_window_tokens: int,
        handoff: bool,
        position: int,
        instances_in_chain: int,
        max_turns: int = 2,
        agent_flavor: str = "bash",
    ):
        import json

        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals import LocalDirStore
        from simple_agent_lab.evals.chain import (
            CHAIN_CONFIG_KEY,
            CHAIN_STATE_INPUT_KEY,
            CHAIN_STATE_OUTPUT_KEY,
            run_chain_in_container,
            state_from_chain_payload,
            state_to_chain_payload,
        )
        from simple_agent_lab.llm.env import FAKE_PROVIDER

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                import subprocess

                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")

            store = LocalDirStore(Path(tmp) / "run")
            prior = start_repo_state("acme/widgets", agent_name="swebench_agent")
            append_instance_task(
                prior,
                agent_name="swebench_agent",
                instance_id="case-0",
                task="prior repo context",
            )
            store.put(
                CHAIN_STATE_INPUT_KEY,
                (
                    json.dumps(state_to_chain_payload(prior), ensure_ascii=False) + "\n"
                ).encode("utf-8"),
            )
            store.put(
                CHAIN_CONFIG_KEY,
                json.dumps(
                    {
                        "mode": "repo_chain",
                        "repo": "acme/widgets",
                        "chain_id": "acme/widgets",
                        "part_index": 1,
                        "part_count": 1,
                        "position": position,
                        "instances_in_chain": instances_in_chain,
                        "provider_auth_env": "OPENAI_AUTH_TOKEN",
                        "config": {
                            "agent_flavor": agent_flavor,
                            "solver_read": False,
                            "task_tool": False,
                            "compression_strategy": "none",
                            "handoff": handoff,
                            "context_window_tokens": context_window_tokens,
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            instance = {
                "instance_id": "case-1",
                "repo": "acme/widgets",
                "problem_statement": (
                    "Change app.py. <bash>printf 'x = 2\\n' > app.py</bash>"
                ),
                "language": "python",
            }

            result, state = run_chain_in_container(
                instance=instance,
                container_module=SWEBENCH_CONTAINER,
                provider=FAKE_PROVIDER,
                workdir=repo,
                max_turns=max_turns,
                store=store,
                trace_id="trace.case-1",
                producer="suite:swebench_pro",
                suite_name="swebench_pro",
                request_extra={},
            )
            restored = state_from_chain_payload(
                json.loads(store.get(CHAIN_STATE_OUTPUT_KEY).decode("utf-8"))
            )
            # `restored` is the OUTGOING chain payload (active context only, what
            # the next window sees); `state` is the full in-container trace.
            return result, restored, state

    def test_mid_instance_handoff_continues_same_instance(self) -> None:
        # Sole instance in the part (so no boundary handoff can fire); a tiny
        # window forces an in-place reset WHILE the instance is still solving.
        result, restored, state = self._run_bash_instance_with_handoff_config(
            context_window_tokens=1,
            handoff=True,
            position=1,
            instances_in_chain=1,
            max_turns=4,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["handoff"])
        self.assertTrue(result["handoff_written"])
        # The window reset mid-instance at least once and kept the SAME instance.
        self.assertGreaterEqual(result["context_window_handoffs"], 1)
        self.assertGreaterEqual(result["chain_window_index"], 2)
        self.assertGreater(result["handoff_context_tokens"], 0)
        # It resumed after the reset and still produced the edit for its own task.
        self.assertIn("app.py", result["model_patch"])
        self.assertIn("x = 2", result["model_patch"])

        # The full in-container trace is preserved (nothing is lost), even though
        # the model's active view was reset.
        full_trace = "\n".join(message_text(message) for message in state.messages)
        self.assertIn("prior repo context", full_trace)
        self.assertIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", full_trace)

        # The window the instance continues in is seeded with the handoff notes.
        visible = "\n".join(
            message_text(message) for message in restored.active_context_messages()
        )
        self.assertIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", visible)

    def test_no_handoff_when_context_stays_within_budget(self) -> None:
        result, restored, _state = self._run_bash_instance_with_handoff_config(
            context_window_tokens=272_000,
            handoff=True,
            position=1,
            instances_in_chain=2,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["handoff"])
        self.assertFalse(result["handoff_written"])
        self.assertEqual(result["context_window_handoffs"], 0)
        self.assertEqual(result["chain_window_index"], 1)

        visible = "\n".join(
            message_text(message) for message in restored.active_context_messages()
        )
        # Under budget, the full chain context carries forward unchanged.
        self.assertNotIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", visible)
        self.assertIn("prior repo context", visible)

    def test_boundary_handoff_starts_next_instance_fresh(self) -> None:
        # One turn per window means the instance completes its turn without a
        # mid-instance abort, but ends over the window: the boundary handoff
        # fires and the NEXT instance starts fresh with only the handoff notes.
        result, restored, _state = self._run_bash_instance_with_handoff_config(
            context_window_tokens=1,
            handoff=True,
            position=1,
            instances_in_chain=2,
            max_turns=1,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["handoff_written"])
        # This is a boundary handoff, not a mid-instance reset.
        self.assertEqual(result["context_window_handoffs"], 0)
        self.assertEqual(result["chain_window_index"], 2)

        visible = "\n".join(
            message_text(message) for message in restored.active_context_messages()
        )
        # The next window is seeded ONLY with the handoff notes; the earlier
        # transcript is intentionally dropped.
        self.assertIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", visible)
        self.assertNotIn("prior repo context", visible)
        self.assertEqual(restored.data["eval_chain"]["window_index"], 2)

    def test_last_instance_resets_window_but_skips_boundary_handoff(self) -> None:
        # The last instance still resets its own window mid-solve (it must stay
        # under the real model window), but there is no next instance to hand off
        # to, so no boundary handoff fires.
        result, restored, state = self._run_bash_instance_with_handoff_config(
            context_window_tokens=1,
            handoff=True,
            position=2,
            instances_in_chain=2,
            max_turns=4,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["handoff_written"])
        self.assertGreaterEqual(result["context_window_handoffs"], 1)
        self.assertGreaterEqual(result["chain_window_index"], 2)

        # Mid-instance reset keeps the SAME state going, so the outgoing window
        # still carries this instance's own task alongside the handoff notes
        # (a boundary handoff would have produced a doc-only fresh state).
        visible = "\n".join(
            message_text(message) for message in restored.active_context_messages()
        )
        self.assertIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", visible)
        full_trace = "\n".join(message_text(message) for message in state.messages)
        self.assertIn("prior repo context", full_trace)

    def test_goal_flavor_resets_window_mid_instance(self) -> None:
        # The goal flavor drives its own loop; verify the mid-instance handoff
        # wraps it too. A tiny env budget keeps the reset count small.
        with patch.dict(
            os.environ,
            {
                "SAL_WORKFLOW_LOOP_MAX_TURNS": "1",
                "SAL_WORKFLOW_WORKER_MAX_TURNS": "2",
            },
        ):
            result, restored, state = self._run_bash_instance_with_handoff_config(
                context_window_tokens=1,
                handoff=True,
                position=1,
                instances_in_chain=1,
                max_turns=4,
                agent_flavor="goal",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["agent_flavor"], "goal")
        self.assertTrue(result["handoff_written"])
        self.assertGreaterEqual(result["context_window_handoffs"], 1)
        self.assertGreaterEqual(result["chain_window_index"], 2)

        full_trace = "\n".join(message_text(message) for message in state.messages)
        self.assertIn("prior repo context", full_trace)
        self.assertIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", full_trace)

    def test_disabled_handoff_runs_naked_over_budget(self) -> None:
        result, restored, _state = self._run_bash_instance_with_handoff_config(
            context_window_tokens=1,
            handoff=False,
            position=1,
            instances_in_chain=2,
        )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["handoff"])
        self.assertFalse(result["handoff_written"])
        self.assertEqual(result["context_window_handoffs"], 0)
        self.assertEqual(result["chain_window_index"], 1)

        visible = "\n".join(
            message_text(message) for message in restored.active_context_messages()
        )
        # No protection: the full transcript carries forward as-is.
        self.assertNotIn("HANDOFF FROM EARLIER IN THIS REPO CHAIN", visible)
        self.assertIn("prior repo context", visible)


class RepoChainTrajectoryOutputTest(unittest.TestCase):
    def test_trajectory_export_is_enabled_by_default_for_repo_chain_runs(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import _maybe_write_trace

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"

            written = _maybe_write_trace(
                argparse.Namespace(),
                path=path,
                state=SimpleNamespace(task="task", events=[], messages=[]),
                trace_id="trace",
                meta={},
            )

            self.assertTrue(written)
            self.assertTrue(path.exists())

    def test_trajectory_export_can_be_disabled_for_large_runs(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import _maybe_write_trace

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"

            written = _maybe_write_trace(
                argparse.Namespace(write_trajectories=False),
                path=path,
                state=SimpleNamespace(task="task", events=[], messages=[]),
                trace_id="trace",
                meta={"repo": "acme/widgets"},
            )

            self.assertFalse(written)
            self.assertFalse(path.exists())

    def test_parser_defaults_to_writing_trajectories(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertTrue(args.write_trajectories)

    def test_parser_can_disable_trajectory_writes(self) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import build_parser

        args = build_parser().parse_args(["--all", "--no-write-trajectories"])

        self.assertFalse(args.write_trajectories)


class RepoChainInvalidPromptHandlingTest(unittest.TestCase):
    def test_invalid_prompt_error_detection_matches_provider_error_shapes(
        self,
    ) -> None:
        from simple_agent_lab.evals.chain import (
            is_invalid_prompt_error,
        )

        self.assertTrue(is_invalid_prompt_error(RuntimeError("invalid_prompt")))
        self.assertTrue(is_invalid_prompt_error(RuntimeError("code=-4321")))
        self.assertFalse(is_invalid_prompt_error(RuntimeError("rate limit")))

    def test_remaining_turn_budget_counts_failed_invalid_prompt_attempts(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from runs.swebench.run_swebench_pro_repo_chains import _remaining_turn_budget

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        state.record_event(TurnStartEvent(agent="swebench_agent"))
        event_start = len(state.events)
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
        )

        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 3), 3)

        state.record_event(TurnStartEvent(agent="swebench_agent"))
        state.record_event(TurnStartEvent(agent="swebench_agent"))

        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 3), 1)
        self.assertEqual(_remaining_turn_budget(state.events[event_start:], 2), 0)

    def test_invalid_prompt_source_is_chain_task_for_current_problem(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            invalid_prompt_source,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
        )

        self.assertEqual(
            invalid_prompt_source(state, item_id="case-1"),
            "chain_task",
        )

    def test_invalid_prompt_source_is_tool_output_for_latest_tool_result(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            invalid_prompt_source,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
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
            invalid_prompt_source(state, item_id="case-1"),
            "tool_output",
        )

    def test_replace_latest_tool_exchange_removes_call_and_result_with_reminder(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            INVALID_PROMPT_TOOL_REMINDER,
            replace_latest_tool_exchange_for_invalid_prompt,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
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
            replace_latest_tool_exchange_for_invalid_prompt(
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
        reminders = [
            message
            for message in state.active_context_messages()
            if message_text(message) == INVALID_PROMPT_TOOL_REMINDER
        ]
        self.assertEqual([message.kind for message in reminders], ["message"])
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
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            repair_active_tool_pairs,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
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

        self.assertTrue(repair_active_tool_pairs(state, agent_name="swebench_agent"))

        self.assertFalse(
            any(
                message_tool_calls(message)
                for message in state.active_context_messages()
            )
        )
        visible = "\n".join(message_text(m) for m in state.active_context_messages())
        self.assertIn("Removed an incomplete tool call/tool result exchange", visible)
        notes = [
            message
            for message in state.active_context_messages()
            if message_text(message).startswith(
                "Removed an incomplete tool call/tool result exchange"
            )
        ]
        self.assertEqual([message.kind for message in notes], ["message"])
        self.assertTrue(
            any(
                isinstance(event, ContextCompressionEvent)
                and event.strategy == "tool-pair-orphan-repair"
                for event in state.events
            )
        )

    def test_rewrite_chain_task_after_skip_removes_bad_problem_from_context(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            drop_chain_task_for_invalid_prompt_skip,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="bad provider-triggering problem statement",
        )

        self.assertTrue(
            drop_chain_task_for_invalid_prompt_skip(
                state,
                agent_name="swebench_agent",
                item_id="case-1",
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
                and event.strategy == "invalid-prompt-chain-task-drop"
                for event in state.events
            )
        )

    def test_end_instance_after_tool_retry_limit_drops_latest_tool_result(
        self,
    ) -> None:
        from evals.swebench.pro_repo_chain import (
            append_instance_task,
            start_repo_state,
        )
        from simple_agent_lab.evals.chain import (
            end_chain_item_after_invalid_prompt_tool_retry_limit,
            replace_latest_tool_exchange_for_invalid_prompt,
        )

        state = start_repo_state("acme/widgets", agent_name="swebench_agent")
        append_instance_task(
            state,
            agent_name="swebench_agent",
            instance_id="case-1",
            task="Solve this repository task.",
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
            replace_latest_tool_exchange_for_invalid_prompt(
                state, agent_name="swebench_agent"
            )
        )

        self.assertTrue(
            end_chain_item_after_invalid_prompt_tool_retry_limit(
                state,
                agent_name="swebench_agent",
                item_id="case-1",
            )
        )

        self.assertEqual(state.active_context_messages(), [])
        self.assertEqual(state.messages[-1].kind, "message")
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


class RepoChainIncrementalPredictionsTest(unittest.TestCase):
    def test_incremental_predictions_file_is_refreshed_after_each_result(
        self,
    ) -> None:
        from runs.swebench.run_swebench_pro_repo_chains import (
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
        from evals.swebench.pro_repo_chain import CommitTimeResolver

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
        from evals.swebench.pro_repo_chain import CommitTimeResolver

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
        from evals.swebench.pro_repo_chain import CommitTimeResolver

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
