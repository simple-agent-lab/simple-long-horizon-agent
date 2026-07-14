from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


class MemoryChainPlanningTest(unittest.TestCase):
    def test_chains_from_manifest_sorts_issues_by_commit_time(self) -> None:
        from evals.swebench.pro_memory_chain import chains_from_manifest

        manifest = {
            "repos": [
                {
                    "repo": "acme/widgets",
                    "chains": [
                        {
                            "chain_id": "acme-1",
                            "issues": [
                                {"instance_id": "b", "commit_time": "2021-02-01"},
                                {"instance_id": "a", "commit_time": "2021-01-01"},
                                {"instance_id": "c", "commit_time": "2021-03-01"},
                            ],
                        }
                    ],
                }
            ]
        }

        chains = chains_from_manifest(manifest)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].chain_id, "acme-1")
        self.assertEqual(chains[0].repo, "acme/widgets")
        self.assertEqual(chains[0].instance_ids, ("a", "b", "c"))

    def test_chains_from_manifest_skips_empty_and_defaults_chain_id(self) -> None:
        from evals.swebench.pro_memory_chain import chains_from_manifest

        manifest = {
            "repos": [
                {
                    "repo": "acme/widgets",
                    "chains": [
                        {"issues": []},
                        {"issues": [{"instance_id": "only"}]},
                    ],
                }
            ]
        }

        chains = chains_from_manifest(manifest)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].instance_ids, ("only",))
        # No chain_id in the file falls back to a stable repo+first-instance id.
        self.assertEqual(chains[0].chain_id, "acme/widgets-only")

    def test_plan_places_chain_instances_and_derives_singletons(self) -> None:
        from evals.swebench.pro_memory_chain import (
            RawIssueChain,
            plan_memory_chains,
        )

        rows = [
            {"instance_id": "a1", "repo": "acme/widgets", "base_commit": "sha-a1"},
            {"instance_id": "a2", "repo": "acme/widgets", "base_commit": "sha-a2"},
            {"instance_id": "a3", "repo": "acme/widgets", "base_commit": "sha-a3"},
            {"instance_id": "solo", "repo": "acme/widgets", "base_commit": "sha-solo"},
            {"instance_id": "b1", "repo": "other/pkg", "base_commit": "sha-b1"},
        ]
        raw_chains = [
            RawIssueChain(
                chain_id="acme-chain",
                repo="acme/widgets",
                instance_ids=("a1", "a2", "a3"),
            ),
            RawIssueChain(chain_id="pkg-chain", repo="other/pkg", instance_ids=("b1",)),
        ]

        plan = plan_memory_chains(rows, raw_chains, memory=True)

        # Longest chain first, then the length-1 chain, then singletons.
        self.assertEqual(
            [chain.chain_id for chain in plan.chains],
            ["acme-chain", "pkg-chain", "solo"],
        )
        acme = plan.chains[0]
        self.assertEqual(acme.instance_ids, ["a1", "a2", "a3"])
        self.assertTrue(acme.memory_enabled)
        self.assertFalse(acme.is_singleton)

        singleton = plan.chains[-1]
        self.assertTrue(singleton.is_singleton)
        self.assertEqual(singleton.chain_id, "solo")
        self.assertFalse(singleton.memory_enabled)  # singleton memory off by default
        self.assertEqual(plan.instance_count, 5)
        self.assertEqual(plan.missing_instance_ids, ())
        self.assertEqual(plan.duplicate_instance_ids, ())

    def test_plan_reports_missing_chain_instances_without_dropping_present_ones(
        self,
    ) -> None:
        from evals.swebench.pro_memory_chain import (
            RawIssueChain,
            plan_memory_chains,
        )

        rows = [{"instance_id": "a1", "repo": "acme/widgets", "base_commit": "x"}]
        raw_chains = [
            RawIssueChain(
                chain_id="acme-chain",
                repo="acme/widgets",
                instance_ids=("a1", "a2-missing"),
            )
        ]

        plan = plan_memory_chains(rows, raw_chains, memory=True)

        self.assertEqual(plan.missing_instance_ids, ("a2-missing",))
        self.assertEqual(len(plan.chains), 1)
        self.assertEqual(plan.chains[0].instance_ids, ["a1"])

    def test_plan_reports_duplicate_instance_ids_across_chains(self) -> None:
        from evals.swebench.pro_memory_chain import (
            RawIssueChain,
            plan_memory_chains,
        )

        rows = [
            {"instance_id": "a1", "repo": "acme/widgets", "base_commit": "x"},
            {"instance_id": "a2", "repo": "acme/widgets", "base_commit": "y"},
        ]
        raw_chains = [
            RawIssueChain(
                chain_id="chain-1", repo="acme/widgets", instance_ids=("a1", "a2")
            ),
            RawIssueChain(
                chain_id="chain-2", repo="acme/widgets", instance_ids=("a2",)
            ),
        ]

        plan = plan_memory_chains(rows, raw_chains, memory=True)

        self.assertEqual(plan.duplicate_instance_ids, ("a2",))
        # a2 stays in the first chain; the second chain drops to empty and is gone.
        self.assertEqual([chain.chain_id for chain in plan.chain_units], ["chain-1"])

    def test_singleton_memory_flag_enables_singleton_namespaces(self) -> None:
        from evals.swebench.pro_memory_chain import plan_memory_chains

        rows = [{"instance_id": "solo", "repo": "acme/widgets", "base_commit": "x"}]

        plan = plan_memory_chains(rows, [], memory=True, singleton_memory=True)

        self.assertTrue(plan.chains[0].is_singleton)
        self.assertTrue(plan.chains[0].memory_enabled)

    def test_memory_master_switch_off_disables_chain_memory(self) -> None:
        from evals.swebench.pro_memory_chain import (
            RawIssueChain,
            plan_memory_chains,
        )

        rows = [
            {"instance_id": "a1", "repo": "acme/widgets", "base_commit": "x"},
            {"instance_id": "a2", "repo": "acme/widgets", "base_commit": "y"},
        ]
        raw_chains = [
            RawIssueChain(
                chain_id="chain-1", repo="acme/widgets", instance_ids=("a1", "a2")
            )
        ]

        plan = plan_memory_chains(rows, raw_chains, memory=False)

        self.assertFalse(plan.chain_units[0].memory_enabled)

    def test_order_chains_longest_first_breaks_ties_deterministically(self) -> None:
        from evals.swebench.pro_memory_chain import (
            MemoryChain,
            order_chains_longest_first,
        )

        chains = [
            MemoryChain("z", "z/repo", ({"instance_id": "z1"},), False),
            MemoryChain(
                "a",
                "a/repo",
                ({"instance_id": "a1"}, {"instance_id": "a2"}),
                True,
            ),
            MemoryChain("a2", "a/repo", ({"instance_id": "x1"},), False),
        ]

        ordered = order_chains_longest_first(chains)

        # Length 2 first; then the two length-1 chains ordered by repo then id.
        self.assertEqual([chain.chain_id for chain in ordered], ["a", "a2", "z"])


class ChainNodesLoadingTest(unittest.TestCase):
    def test_chains_from_nodes_groups_and_orders_by_step_index(self) -> None:
        from evals.swebench.pro_memory_chain import chains_from_nodes

        # Deliberately out of order: step_index must drive intra-chain order,
        # and chains are grouped by chain_id in first-seen order.
        nodes = [
            {"chain_id": "c1", "step_index": 2, "instance_id": "b", "repo": "r/one"},
            {"chain_id": "c2", "step_index": 1, "instance_id": "x", "repo": "r/two"},
            {"chain_id": "c1", "step_index": 1, "instance_id": "a", "repo": "r/one"},
            {"chain_id": "c1", "step_index": 3, "instance_id": "c", "repo": "r/one"},
        ]

        chains = chains_from_nodes(nodes)

        self.assertEqual([c.chain_id for c in chains], ["c1", "c2"])
        self.assertEqual(chains[0].instance_ids, ("a", "b", "c"))
        self.assertEqual(chains[0].repo, "r/one")
        self.assertEqual(chains[1].instance_ids, ("x",))

    def test_chains_from_nodes_falls_back_to_commit_time_without_step(self) -> None:
        from evals.swebench.pro_memory_chain import chains_from_nodes

        nodes = [
            {"chain_id": "c1", "instance_id": "late", "commit_time": "2021-05-01"},
            {"chain_id": "c1", "instance_id": "early", "commit_time": "2021-01-01"},
        ]

        chains = chains_from_nodes(nodes)

        self.assertEqual(chains[0].instance_ids, ("early", "late"))

    def test_chains_from_nodes_defaults_chain_id_and_skips_idless_nodes(self) -> None:
        from evals.swebench.pro_memory_chain import chains_from_nodes

        nodes = [
            {"step_index": 1, "instance_id": "only", "repo": "r/one"},
            {"chain_id": "c1", "step_index": 1, "instance_id": ""},
        ]

        chains = chains_from_nodes(nodes)

        # The id-less node is dropped; the empty chain_id falls back to repo+id.
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].instance_ids, ("only",))
        self.assertEqual(chains[0].chain_id, "r/one-only")

    def test_load_issue_chains_reads_jsonl_node_file(self) -> None:
        from evals.swebench.pro_memory_chain import load_issue_chains

        nodes = [
            {"chain_id": "c1", "step_index": 1, "instance_id": "a", "repo": "r/one"},
            {"chain_id": "c1", "step_index": 2, "instance_id": "b", "repo": "r/one"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodes.jsonl"
            path.write_text(
                "\n".join(json.dumps(node) for node in nodes) + "\n",
                encoding="utf-8",
            )

            chains = load_issue_chains(path)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].chain_id, "c1")
        self.assertEqual(chains[0].instance_ids, ("a", "b"))

    def test_load_issue_chains_still_reads_nested_json_file(self) -> None:
        from evals.swebench.pro_memory_chain import load_issue_chains

        manifest = {
            "repos": [
                {
                    "repo": "acme/widgets",
                    "chains": [
                        {
                            "chain_id": "acme-1",
                            "issues": [
                                {"instance_id": "a", "commit_time": "2021-01-01"},
                                {"instance_id": "b", "commit_time": "2021-02-01"},
                            ],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chains.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            chains = load_issue_chains(path)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].chain_id, "acme-1")
        self.assertEqual(chains[0].instance_ids, ("a", "b"))

    def test_default_chains_json_is_vendored_in_repo(self) -> None:
        from evals.swebench.pro_memory_chain import (
            DEFAULT_CHAINS_JSON,
            load_issue_chains,
        )

        default_path = Path(DEFAULT_CHAINS_JSON)
        self.assertTrue(
            default_path.exists(),
            f"vendored chains file missing: {default_path}",
        )
        self.assertEqual(default_path.suffix, ".jsonl")
        self.assertEqual(default_path.parent.name, "data")
        # The vendored deep file is the 47-chain analysis; parse it for real.
        chains = load_issue_chains(default_path)
        self.assertEqual(len(chains), 47)
        self.assertEqual(sum(len(c.instance_ids) for c in chains), 261)


class AuthSlotExpansionTest(unittest.TestCase):
    def test_expand_auth_slots_expands_counts(self) -> None:
        from evals.swebench.pro_memory_chain import expand_auth_slots

        slots = expand_auth_slots(
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11",
            default_env="OPENAI_AUTH_TOKEN",
        )

        self.assertEqual(slots[:12], ["OPENAI_AUTH_TOKEN"] * 12)
        self.assertEqual(slots[12:], ["OPENAI_AUTH_TOKEN2"] * 11)

    def test_expand_auth_slots_defaults_to_single_slot(self) -> None:
        from evals.swebench.pro_memory_chain import expand_auth_slots

        self.assertEqual(
            expand_auth_slots(None, default_env="OPENAI_AUTH_TOKEN"),
            ["OPENAI_AUTH_TOKEN"],
        )

    def test_expand_auth_slots_rejects_bad_specs(self) -> None:
        from evals.swebench.pro_memory_chain import expand_auth_slots

        with self.assertRaises(ValueError):
            expand_auth_slots("OPENAI_AUTH_TOKEN", default_env="OPENAI_AUTH_TOKEN")
        with self.assertRaises(ValueError):
            expand_auth_slots("OPENAI_AUTH_TOKEN:0", default_env="OPENAI_AUTH_TOKEN")
        with self.assertRaises(ValueError):
            expand_auth_slots("1BAD:2", default_env="OPENAI_AUTH_TOKEN")

    def test_lane_auth_slots_cycles_and_truncates(self) -> None:
        from evals.swebench.pro_memory_chain import lane_auth_slots

        expanded = ["A", "A", "B"]
        self.assertEqual(lane_auth_slots(expanded, 2), ["A", "A"])
        self.assertEqual(lane_auth_slots(expanded, 5), ["A", "A", "B", "A", "A"])


class ModelNameTest(unittest.TestCase):
    def test_model_name_reflects_memory_arm(self) -> None:
        from evals.swebench.pro_memory_chain import model_name_for_config

        self.assertEqual(
            model_name_for_config(
                agent_flavor="bash", memory=True, singleton_memory=False
            ),
            "simple-agent-lab-pro-memory-chain-bash-memory",
        )
        self.assertEqual(
            model_name_for_config(
                agent_flavor="bash", memory=True, singleton_memory=True
            ),
            "simple-agent-lab-pro-memory-chain-bash-memory-all",
        )
        self.assertEqual(
            model_name_for_config(
                agent_flavor="bash_task", memory=False, singleton_memory=False
            ),
            "simple-agent-lab-pro-memory-chain-bash_task-nomemory",
        )


class PlanManifestTest(unittest.TestCase):
    def test_manifest_counts_and_histogram(self) -> None:
        from evals.swebench.pro_memory_chain import (
            ProMemoryChainConfig,
            RawIssueChain,
            plan_manifest,
            plan_memory_chains,
        )

        rows = [
            {"instance_id": "a1", "repo": "acme/widgets", "base_commit": "1"},
            {"instance_id": "a2", "repo": "acme/widgets", "base_commit": "2"},
            {"instance_id": "solo", "repo": "acme/widgets", "base_commit": "3"},
        ]
        raw_chains = [
            RawIssueChain(
                chain_id="chain-1", repo="acme/widgets", instance_ids=("a1", "a2")
            )
        ]
        plan = plan_memory_chains(rows, raw_chains, memory=True)

        manifest = plan_manifest(
            plan, config=ProMemoryChainConfig(), run_id="run-1", parallel=4
        )

        self.assertEqual(manifest["run_unit_count"], 2)
        self.assertEqual(manifest["chain_count"], 1)
        self.assertEqual(manifest["singleton_count"], 1)
        self.assertEqual(manifest["instance_count"], 3)
        self.assertEqual(manifest["chain_instance_count"], 2)
        self.assertEqual(manifest["chain_length_histogram"], {"2": 1})
        self.assertEqual(manifest["parallel"], 4)
        self.assertEqual(manifest["order"][0]["chain_id"], "chain-1")


class MemoryChainRunnerParserTest(unittest.TestCase):
    def test_runner_defaults(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import build_parser

        args = build_parser().parse_args(["--all"])

        self.assertEqual(args.api_kind, "openai-responses")
        self.assertEqual(args.agent_flavor, "bash")
        self.assertTrue(args.memory)
        self.assertFalse(args.singleton_memory)
        self.assertEqual(args.parallel, "slots")
        self.assertIsNone(args.chains_json)
        self.assertIsNone(args.model)
        self.assertIsNone(args.provider_auth_envs)

    def test_runner_requires_explicit_chains_json_to_load(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _load_chains,
            build_parser,
        )

        args = build_parser().parse_args(["--all"])

        with self.assertRaises(SystemExit) as raised:
            _load_chains(args)

        self.assertIn("Pass --chains-json PATH", str(raised.exception))

    def test_runner_rejects_workflow_flavor(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import build_parser

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            build_parser().parse_args(["--all", "--agent-flavor", "goal"])

    def test_experiment_config_records_env_model_and_memory_arm(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _apply_provider_env_overrides,
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all", "--no-memory"])

        with patch.dict(
            os.environ,
            {"OPENAI_MODEL": "env-model", "REASONING_EFFORT": "high"},
            clear=True,
        ):
            _apply_provider_env_overrides(args)
            config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.model, "env-model")
        self.assertEqual(config.reasoning_effort, "high")
        self.assertFalse(config.memory)
        self.assertEqual(
            config.model_name, "simple-agent-lab-pro-memory-chain-bash-nomemory"
        )

    def test_bash_task_flavor_sets_task_tool_flag(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _experiment_config_from_args,
            build_parser,
        )

        args = build_parser().parse_args(["--all", "--agent-flavor", "bash_task"])
        config = _experiment_config_from_args(args, api_kind="openai-responses")

        self.assertEqual(config.agent_flavor, "bash_task")
        self.assertTrue(config.task_tool)

    def test_default_run_id_is_derived_from_arm(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _experiment_config_from_args,
            _resolve_run_id,
            build_parser,
        )

        now = datetime(2026, 7, 8, 21, 30, 0)
        memory_args = build_parser().parse_args(["--all"])
        memory_config = _experiment_config_from_args(
            memory_args, api_kind="openai-responses"
        )
        nomem_args = build_parser().parse_args(["--all", "--no-memory"])
        nomem_config = _experiment_config_from_args(
            nomem_args, api_kind="openai-responses"
        )

        self.assertEqual(
            _resolve_run_id(None, memory_config, now=now),
            "pro-memory-chain-memory-20260708-213000",
        )
        self.assertEqual(
            _resolve_run_id(None, nomem_config, now=now),
            "pro-memory-chain-nomemory-20260708-213000",
        )
        self.assertEqual(_resolve_run_id("manual", memory_config, now=now), "manual")

    def test_resolve_parallel_from_slots_or_int(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import _resolve_parallel

        self.assertEqual(_resolve_parallel("slots", slot_count=23), 23)
        self.assertEqual(_resolve_parallel("8", slot_count=23), 8)
        with self.assertRaises(SystemExit):
            _resolve_parallel("0", slot_count=23)

    def test_provider_env_sets_flavor_and_memory_scope(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _provider_env_for_instance,
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "model",
                "OPENAI_AUTH_TOKEN2": "token-2",
                "REASONING_EFFORT": "high",
            },
            clear=True,
        ):
            env = _provider_env_for_instance(
                "OPENAI_AUTH_TOKEN2",
                api_kind="openai-responses",
                agent_flavor="bash",
                memory_name="acme-chain",
                memory_run_id="001_instance_a1",
            )

        self.assertEqual(env["OPENAI_AUTH_TOKEN"], "token-2")
        self.assertEqual(env["OPENAI_MODEL"], "model")
        self.assertEqual(env["API_KIND"], "openai-responses")
        self.assertEqual(env["AGENT_FLAVOR"], "bash")
        self.assertEqual(env["SAL_MEMORY_NAME"], "acme-chain")
        self.assertEqual(env["SAL_MEMORY_RUN_ID"], "001_instance_a1")

    def test_provider_env_omits_memory_scope_for_singletons(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _provider_env_for_instance,
        )

        with patch.dict(
            os.environ,
            {"OPENAI_MODEL": "model", "OPENAI_AUTH_TOKEN": "token-1"},
            clear=True,
        ):
            env = _provider_env_for_instance(
                "OPENAI_AUTH_TOKEN",
                api_kind="openai-responses",
                agent_flavor="bash",
                memory_name="",
                memory_run_id="001_solo",
            )

        self.assertNotIn("SAL_MEMORY_NAME", env)
        self.assertNotIn("SAL_MEMORY_RUN_ID", env)

    def test_provider_env_requires_model_and_token(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _provider_env_for_instance,
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(SystemExit),
        ):
            _provider_env_for_instance(
                "OPENAI_AUTH_TOKEN",
                api_kind="openai-responses",
                agent_flavor="bash",
                memory_name="",
                memory_run_id="x",
            )

    def test_expand_auth_slots_wrapper_raises_system_exit(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import _expand_auth_slots

        with self.assertRaises(SystemExit):
            _expand_auth_slots("OPENAI_AUTH_TOKEN:0")

    def test_select_units_applies_max_chains_and_limit(self) -> None:
        from evals.swebench.pro_memory_chain import (
            ProMemoryChainConfig,
            RawIssueChain,
            plan_manifest,
            plan_memory_chains,
        )
        from runs.swebench.run_swebench_pro_memory_chains import (
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
        plan = plan_memory_chains(rows, raw_chains, memory=True)

        max_chain_args = build_parser().parse_args(["--all", "--max-chains", "1"])
        kept = _select_units(plan.chains, max_chain_args)
        self.assertEqual(
            [unit.chain_id for unit in kept if not unit.is_singleton], ["c-a"]
        )
        # Singletons are still kept when only chains are capped.
        self.assertTrue(any(unit.is_singleton for unit in kept))

        limit_args = build_parser().parse_args(["--all", "--limit", "2"])
        limited = _select_units(plan.chains, limit_args)
        self.assertEqual(sum(unit.length for unit in limited), 2)
        manifest = plan_manifest(
            plan,
            config=ProMemoryChainConfig(),
            run_id="limited",
            parallel=1,
            run_units=limited,
        )
        self.assertEqual(manifest["instance_count"], 2)
        self.assertEqual(manifest["run_unit_count"], len(limited))
        self.assertEqual(
            [entry["chain_id"] for entry in manifest["order"]],
            [unit.chain_id for unit in limited],
        )


class MemoryChainIncrementalPredictionsTest(unittest.TestCase):
    def test_incremental_predictions_include_unfinished_planned_instances(self) -> None:
        from runs.swebench.run_swebench_pro_memory_chains import (
            _write_incremental_predictions,
        )
        from simple_agent_lab.trace import read_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "run-1"
            instance_dir = root / run_id / "instance_one"
            (instance_dir / "input").mkdir(parents=True)
            (instance_dir / "out").mkdir()
            (instance_dir / "input" / "instance.json").write_text(
                '{"instance_id": "instance_one"}\n', encoding="utf-8"
            )
            (instance_dir / "out" / "result.json").write_text(
                '{"model_patch": "diff --git a/a b/a\\n"}\n', encoding="utf-8"
            )
            predictions_path = root / run_id / "run-1_predictions.jsonl"

            _write_incremental_predictions(
                predictions_path=predictions_path,
                run_root=root,
                run_id=run_id,
                model_name="model",
                dataset_name="ScaleAI/SWE-bench_Pro",
                expected_instance_ids=("instance_one", "instance_two"),
                lock=None,
            )

            predictions = read_jsonl(predictions_path)
            self.assertEqual(
                [prediction["instance_id"] for prediction in predictions],
                ["instance_one", "instance_two"],
            )
            self.assertEqual(predictions[1]["patch"], "")


if __name__ == "__main__":
    unittest.main()
