from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from simple_agent_lab.evals import (
    DatasetReport,
    InstanceResult,
    RESULT_KEY,
    RunArtifacts,
)

from runs import run_gdpval


class RunGdpvalJudgeRetryTest(unittest.TestCase):
    def test_default_image_uses_boyuan_gdpval_base(self) -> None:
        args = run_gdpval._build_parser().parse_args([])

        self.assertEqual(
            args.image,
            "hub.byted.org/boyuan/gdpval-agent-base:latest",
        )
        self.assertEqual(args.judge_tool_mode, "hybrid")

    def test_provider_env_includes_azure_openai_settings(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "OPENAI_MODEL": "gpt-test",
                "OPENAI_AUTH_TOKEN": "token",
                "AZURE_OPENAI_ENDPOINT": "https://azure.example.test",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_OPENAI_LOGID": "log-123",
            },
            clear=True,
        ):
            self.assertEqual(
                run_gdpval._provider_env(),
                {
                    "OPENAI_MODEL": "gpt-test",
                    "OPENAI_AUTH_TOKEN": "token",
                    "AZURE_OPENAI_ENDPOINT": "https://azure.example.test",
                    "AZURE_OPENAI_API_VERSION": "2024-02-01",
                    "AZURE_OPENAI_LOGID": "log-123",
                },
            )

    def test_solver_and_judge_provider_env_can_differ_from_cli(self) -> None:
        args = run_gdpval._build_parser().parse_args(
            [
                "--solver-model",
                "solver-model",
                "--solver-api-key",
                "solver-key",
                "--solver-base-url",
                "https://solver.example/v1",
                "--judge-model",
                "judge-model",
                "--judge-api-key",
                "judge-key",
                "--judge-base-url",
                "https://judge.example/v1",
            ]
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            solver_env = run_gdpval._provider_env(args, stage="solver")
            judge_env = run_gdpval._provider_env(args, stage="judge", base=solver_env)

        self.assertEqual(
            solver_env,
            {
                "OPENAI_MODEL": "solver-model",
                "OPENAI_AUTH_TOKEN": "solver-key",
                "OPENAI_BASE_URL": "https://solver.example/v1",
            },
        )
        self.assertEqual(
            judge_env,
            {
                "OPENAI_MODEL": "judge-model",
                "OPENAI_AUTH_TOKEN": "judge-key",
                "OPENAI_BASE_URL": "https://judge.example/v1",
            },
        )

    def test_judge_prefixed_env_overrides_solver_env_for_azure(self) -> None:
        args = run_gdpval._build_parser().parse_args(
            ["--solver-base-url", "https://solver.example/v1"]
        )
        with mock.patch.dict(
            "os.environ",
            {
                "OPENAI_MODEL": "solver-model",
                "OPENAI_AUTH_TOKEN": "solver-key",
                "JUDGE_OPENAI_MODEL": "judge-model",
                "JUDGE_OPENAI_AUTH_TOKEN": "judge-key",
                "JUDGE_AZURE_OPENAI_ENDPOINT": "https://judge.azure.test",
                "JUDGE_AZURE_OPENAI_API_VERSION": "2024-02-01",
                "JUDGE_AZURE_OPENAI_LOGID": "judge-log",
            },
            clear=True,
        ):
            solver_env = run_gdpval._provider_env(args, stage="solver")
            judge_env = run_gdpval._provider_env(args, stage="judge", base=solver_env)

        self.assertEqual(solver_env["OPENAI_MODEL"], "solver-model")
        self.assertEqual(solver_env["OPENAI_BASE_URL"], "https://solver.example/v1")
        self.assertEqual(judge_env["OPENAI_MODEL"], "judge-model")
        self.assertEqual(judge_env["OPENAI_AUTH_TOKEN"], "judge-key")
        self.assertNotIn("OPENAI_BASE_URL", judge_env)
        self.assertEqual(judge_env["AZURE_OPENAI_ENDPOINT"], "https://judge.azure.test")
        self.assertEqual(judge_env["AZURE_OPENAI_API_VERSION"], "2024-02-01")
        self.assertEqual(judge_env["AZURE_OPENAI_LOGID"], "judge-log")

    def test_semantic_retry_only_reruns_invalid_judge_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[tuple[str, list[str]]] = []

            def result(run_id: str, task_id: str, status: str, score: float):
                run_dir = root / run_id / task_id
                out_dir = run_dir / "out"
                out_dir.mkdir(parents=True)
                (run_dir / RESULT_KEY).write_text(
                    json.dumps(
                        {
                            "status": status,
                            "score": score,
                            "earned_score": score,
                            "max_score": 1.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return InstanceResult(
                    task_id,
                    RunArtifacts(
                        instance_id=task_id,
                        run_dir=run_dir,
                        trajectory_path=out_dir / "trajectory.jsonl",
                        status_code=0,
                    ),
                    None,
                    attempts=1,
                )

            def fake_run_dataset(**kwargs):
                run_id = kwargs["run_id"]
                ids = [str(item["instance_id"]) for item in kwargs["instances"]]
                calls.append((run_id, ids))
                if run_id == "judge-base":
                    return DatasetReport(
                        [
                            result("judge-base", "good", "gsb_judged", 0.5),
                            result(
                                "judge-base",
                                "flaky",
                                "judge_result_missing",
                                0.0,
                            ),
                        ]
                    )
                self.assertEqual(run_id, "judge-base-semantic-retry-2")
                self.assertEqual(ids, ["flaky"])
                return DatasetReport([result(run_id, "flaky", "gsb_judged", 1.0)])

            args = argparse.Namespace(
                judge_semantic_max_attempts=2,
                judge_concurrency=None,
                concurrency=1,
                judge_max_attempts=1,
                judge_max_turns=50,
                wheelhouse_mount="/agent/wheelhouse",
            )

            with (
                mock.patch.object(
                    run_gdpval, "run_dataset", side_effect=fake_run_dataset
                ),
                mock.patch.object(run_gdpval, "_backend_for", return_value=object()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results, run_ids, attempt_counts, histories = (
                    run_gdpval._run_judge_with_semantic_retries(
                        args=args,
                        suite=object(),
                        judge_instances=[
                            {"instance_id": "good"},
                            {"instance_id": "flaky"},
                        ],
                        run_root=root,
                        base_judge_run_id="judge-base",
                        wheelhouse=None,
                        judge_provider="fake",
                        judge_api_kind="openai-chat",
                        provider_env={},
                        on_judge_result=lambda result: None,
                    )
                )

            self.assertEqual(
                calls,
                [
                    ("judge-base", ["good", "flaky"]),
                    ("judge-base-semantic-retry-2", ["flaky"]),
                ],
            )
            self.assertEqual([item.instance_id for item in results], ["good", "flaky"])
            self.assertEqual(run_ids, ["judge-base", "judge-base-semantic-retry-2"])
            self.assertEqual(attempt_counts, {"good": 1, "flaky": 2})
            self.assertEqual(
                histories,
                {
                    "good": ["gsb_judged"],
                    "flaky": ["judge_result_missing", "gsb_judged"],
                },
            )

            summary = run_gdpval._write_judge_summary(
                run_root=root,
                solver_run_id="solver",
                judge_run_id="judge-base",
                judge_run_ids=run_ids,
                judge_mode="gsb",
                results=results,
                skipped=[],
                attempt_counts=attempt_counts,
                semantic_histories=histories,
            )
            self.assertEqual(summary["judged"], 2)
            self.assertEqual(summary["mean_score"], 0.75)

            payload = json.loads((root / "solver" / "judge_summary.json").read_text())
            by_id = {row["task_id"]: row for row in payload["rows"]}
            self.assertEqual(by_id["flaky"]["status"], "gsb_judged")
            self.assertEqual(by_id["flaky"]["semantic_attempts"], 2)
            self.assertEqual(
                by_id["flaky"]["semantic_status_history"],
                ["judge_result_missing", "gsb_judged"],
            )


if __name__ == "__main__":
    unittest.main()
