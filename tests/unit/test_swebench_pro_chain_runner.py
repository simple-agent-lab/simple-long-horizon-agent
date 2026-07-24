"""Unit tests for shared SWE-bench Pro chain-runner plumbing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.swebench import pro_chain_runner
from evals.swebench.pro_memory_chain import (
    SINGLETON_CHAIN_SOURCE,
    MemoryChain,
)


def _unit(
    chain_id: str,
    repo: str,
    instance_ids: tuple[str, ...],
    *,
    singleton: bool = False,
) -> MemoryChain:
    return MemoryChain(
        chain_id=chain_id,
        repo=repo,
        rows=tuple(
            {"instance_id": instance_id, "repo": repo} for instance_id in instance_ids
        ),
        memory_enabled=not singleton,
        source=SINGLETON_CHAIN_SOURCE if singleton else "chain",
    )


class ProChainRunnerTest(unittest.TestCase):
    def test_resolve_auth_lanes_preserves_declared_slot_order(self) -> None:
        lanes = pro_chain_runner.resolve_auth_lanes("TOKEN_A:2,TOKEN_B:1", "slots")

        self.assertEqual(lanes.parallel, 3)
        self.assertEqual(lanes.slots, ("TOKEN_A", "TOKEN_A", "TOKEN_B"))
        self.assertEqual(
            lanes.as_manifest(),
            {
                "spec": "TOKEN_A:2,TOKEN_B:1",
                "lane_slots": ["TOKEN_A", "TOKEN_A", "TOKEN_B"],
            },
        )

    def test_select_units_keeps_singletons_when_limiting_chains(self) -> None:
        units = (
            _unit("long", "repo-a", ("a1", "a2", "a3")),
            _unit("short", "repo-a", ("a4", "a5")),
            _unit("solo", "repo-b", ("b1",), singleton=True),
        )

        selected = pro_chain_runner.select_units(
            units,
            max_chains=1,
            limit=None,
            limit_per_repo=2,
        )

        self.assertEqual([unit.chain_id for unit in selected], ["long", "solo"])
        self.assertEqual(selected[0].instance_ids, ["a1", "a2"])
        self.assertEqual(selected[1].instance_ids, ["b1"])

    def test_auth_slot_is_returned_after_worker_failure(self) -> None:
        seen: list[tuple[str, str]] = []
        completed: list[str] = []
        lanes = pro_chain_runner.AuthLanes(
            parallel=1,
            slots=("TOKEN_A",),
            spec="TOKEN_A:1",
        )

        def worker(unit: str, auth_env: str) -> dict[str, object]:
            seen.append((unit, auth_env))
            if unit == "bad":
                raise RuntimeError("boom")
            return {"ok": True}

        failures = pro_chain_runner.run_auth_lanes(
            ("bad", "good"),
            lanes=lanes,
            chain_id=str,
            worker=worker,
            on_done=lambda unit_id, _result: completed.append(unit_id),
        )

        self.assertEqual(seen, [("bad", "TOKEN_A"), ("good", "TOKEN_A")])
        self.assertEqual(completed, ["good"])
        self.assertEqual(failures, [{"chain_id": "bad", "error": "RuntimeError: boom"}])

    def test_provider_env_maps_selected_auth_slot_to_container_contract(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": " model ",
                "TOKEN_B": " token ",
                "OPENAI_BASE_URL": " https://example.test ",
                "NO_PROXY": " localhost,127.0.0.1 ",
            },
            clear=True,
        ):
            env = pro_chain_runner.provider_env_for_auth_env(
                "TOKEN_B", api_kind="openai-responses"
            )

        self.assertEqual(env["OPENAI_MODEL"], "model")
        self.assertEqual(env["OPENAI_AUTH_TOKEN"], "token")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://example.test")
        self.assertEqual(env["NO_PROXY"], " localhost,127.0.0.1 ")
        self.assertEqual(env["API_KIND"], "openai-responses")
        self.assertNotIn("TOKEN_B", env)

    def test_provider_env_reports_selected_missing_auth_slot(self) -> None:
        with (
            mock.patch.dict(os.environ, {"OPENAI_MODEL": "model"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "TOKEN_B"),
        ):
            pro_chain_runner.provider_env_for_auth_env(
                "TOKEN_B", api_kind="openai-responses"
            )

    def test_batch_output_retains_full_expected_denominator(self) -> None:
        rows = [
            {"instance_id": "first", "repo": "repo"},
            {"instance_id": "second", "repo": "repo"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            output = pro_chain_runner.prepare_batch_output(
                run_root=run_root,
                run_id="run-1",
                rows=rows,
                manifest={"schema": "test.v1"},
            )
            with mock.patch.object(
                pro_chain_runner,
                "predictions_from_run_dirs",
                return_value=[],
            ) as predictions:
                pro_chain_runner.write_predictions(
                    output,
                    model_name="model",
                    dataset_name="dataset",
                )

            predictions.assert_called_once_with(
                run_root,
                run_id="run-1",
                model_name="model",
                dataset_name="dataset",
                expected_instance_ids=("first", "second"),
            )
            self.assertEqual(
                output.predictions_path,
                output.batch_dir / "run-1_predictions.jsonl",
            )
            self.assertEqual(
                json.loads(
                    (output.batch_dir / "experiment.json").read_text(encoding="utf-8")
                ),
                {"schema": "test.v1"},
            )
            self.assertEqual(
                [
                    json.loads(line)
                    for line in output.instances_json.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ],
                rows,
            )

    def test_official_eval_uses_pro_instances_and_resolved_parallelism(self) -> None:
        predictions = Path("/tmp/predictions.jsonl")
        instances = Path("/tmp/instances.jsonl")
        with mock.patch.object(pro_chain_runner.subprocess, "run") as run:
            pro_chain_runner.run_official_eval(
                predictions_path=predictions,
                instances_json=instances,
                run_id="run-1",
                max_workers=7,
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[1:4],
            [
                str(pro_chain_runner.ROOT / "evals/swebench/evaluate_predictions.py"),
                "--pro",
                "--run-official",
            ],
        )
        self.assertEqual(command[command.index("--predictions") + 1], str(predictions))
        self.assertEqual(command[command.index("--instances") + 1], str(instances))
        self.assertEqual(command[command.index("--run-id") + 1], "run-1")
        self.assertEqual(command[command.index("--max-workers") + 1], "7")
        self.assertEqual(
            run.call_args.kwargs,
            {"cwd": pro_chain_runner.ROOT, "check": True},
        )
