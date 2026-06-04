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
