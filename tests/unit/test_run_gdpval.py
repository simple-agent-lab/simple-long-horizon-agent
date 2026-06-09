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
    InstanceResult,
    RESULT_KEY,
    RunArtifacts,
)

from runs import run_gdpval
from runs import run_gdpval_judge_existing


class RunGdpvalJudgeRetryTest(unittest.TestCase):
    def test_default_image_uses_boyuan_gdpval_base(self) -> None:
        args = run_gdpval._build_parser().parse_args([])

        self.assertEqual(
            args.image,
            "hub.byted.org/boyuan/gdpval-agent-base:latest",
        )
        self.assertEqual(args.judge_tool_mode, "hybrid")
        self.assertFalse(args.include_known_bad_tasks)
        self.assertTrue(args.enable_web_tools)

    def test_disable_web_tools_flag(self) -> None:
        args = run_gdpval._build_parser().parse_args(["--disable-web-tools"])

        self.assertFalse(args.enable_web_tools)

    def test_candidate_missing_judge_status_is_semantic_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "judge" / "missing"
            out_dir = run_dir / "out"
            out_dir.mkdir(parents=True)
            (run_dir / RESULT_KEY).write_text(
                json.dumps(
                    {
                        "status": "candidate_deliverables_missing",
                        "score": 0.0,
                        "earned_score": 0.0,
                        "max_score": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = InstanceResult(
                "missing",
                RunArtifacts(
                    instance_id="missing",
                    run_dir=run_dir,
                    trajectory_path=out_dir / "trajectory.jsonl",
                    status_code=0,
                ),
                None,
                attempts=1,
            )

            self.assertTrue(run_gdpval._judge_result_is_semantic_success(item))

    def test_provider_env_includes_azure_openai_settings(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "OPENAI_MODEL": "gpt-test",
                "OPENAI_AUTH_TOKEN": "token",
                "OPENAI_REASONING_EFFORT": "high",
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
                    "OPENAI_REASONING_EFFORT": "high",
                    "AZURE_OPENAI_ENDPOINT": "https://azure.example.test",
                    "AZURE_OPENAI_API_VERSION": "2024-02-01",
                    "AZURE_OPENAI_LOGID": "log-123",
                },
            )

    def test_web_tool_env_collects_serper_and_jina_only(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "SERPER_API_KEY": "serper-key",
                "JINA_API_KEY": "jina-key",
                "JINA_ENDPOINT": "https://r.jina.ai/",
                "PROXY": "should-not-pass",
            },
            clear=True,
        ):
            self.assertEqual(
                run_gdpval._web_tool_env(),
                {
                    "SERPER_API_KEY": "serper-key",
                    "JINA_API_KEY": "jina-key",
                    "JINA_ENDPOINT": "https://r.jina.ai/",
                },
            )

    def test_existing_judge_responses_env_keeps_session_and_normalizes_base_url(
        self,
    ) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "OPENAI_MODEL": "judge-model",
                "OPENAI_AUTH_TOKEN": "token",
                "OPENAI_BASE_URL": "https://example.test/api/responses",
                "OPENAI_SESSION_ID": "judge-session",
                "OPENAI_REASONING_EFFORT": "high",
            },
            clear=True,
        ):
            env = run_gdpval_judge_existing._judge_provider_env(
                judge_api_kind="openai-responses"
            )

        self.assertEqual(env["OPENAI_MODEL"], "judge-model")
        self.assertEqual(env["OPENAI_AUTH_TOKEN"], "token")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://example.test/api")
        self.assertEqual(env["OPENAI_SESSION_ID"], "judge-session")
        self.assertEqual(env["OPENAI_REASONING_EFFORT"], "high")

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

    def test_semantic_retry_streams_invalid_judge_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[tuple[str, str]] = []

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

            def fake_run_suite_instance(**kwargs):
                run_id = kwargs["run_id"]
                task_id = str(kwargs["instance"]["instance_id"])
                calls.append((run_id, task_id))
                if run_id == "judge-base" and task_id == "flaky":
                    return result(
                        run_id, task_id, "judge_result_missing", 0.0
                    ).artifacts
                score = 1.0 if task_id == "flaky" else 0.5
                return result(run_id, task_id, "gsb_judged", score).artifacts

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
                    run_gdpval,
                    "run_suite_instance",
                    side_effect=fake_run_suite_instance,
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
                            {"instance_id": "later"},
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
                    ("judge-base", "good"),
                    ("judge-base", "flaky"),
                    ("judge-base-semantic-retry-2", "flaky"),
                    ("judge-base", "later"),
                ],
            )
            self.assertEqual(
                [item.instance_id for item in results],
                ["good", "flaky", "later"],
            )
            self.assertEqual(run_ids, ["judge-base", "judge-base-semantic-retry-2"])
            self.assertEqual(attempt_counts, {"good": 1, "flaky": 2, "later": 1})
            self.assertEqual(
                histories,
                {
                    "good": ["gsb_judged"],
                    "flaky": ["judge_result_missing", "gsb_judged"],
                    "later": ["gsb_judged"],
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
            self.assertEqual(summary["judged"], 3)
            self.assertEqual(summary["mean_score"], 2.0 / 3.0)

            payload = json.loads((root / "solver" / "judge_summary.json").read_text())
            by_id = {row["task_id"]: row for row in payload["rows"]}
            self.assertEqual(by_id["flaky"]["status"], "gsb_judged")
            self.assertEqual(by_id["flaky"]["semantic_attempts"], 2)
            self.assertEqual(
                by_id["flaky"]["semantic_status_history"],
                ["judge_result_missing", "gsb_judged"],
            )

    def test_streaming_judge_starts_from_solver_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver_run_dir = root / "solver" / "ready"
            solver_out = solver_run_dir / "out"
            solver_out.mkdir(parents=True)
            (solver_run_dir / RESULT_KEY).write_text(
                json.dumps({"status": "solver_finished"}) + "\n",
                encoding="utf-8",
            )
            solver_result = InstanceResult(
                "ready",
                RunArtifacts(
                    instance_id="ready",
                    run_dir=solver_run_dir,
                    trajectory_path=solver_out / "trajectory.jsonl",
                    status_code=0,
                ),
                None,
                attempts=1,
            )
            judge_run_dir = root / "judge-base" / "ready"
            judge_out = judge_run_dir / "out"
            judge_out.mkdir(parents=True)
            (judge_run_dir / RESULT_KEY).write_text(
                json.dumps(
                    {
                        "status": "gsb_judged",
                        "score": 1.0,
                        "earned_score": 1.0,
                        "max_score": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            started: list[str] = []

            class FakeSuite:
                def build_instance(self, source, *, candidate_result, candidate_artifacts):
                    started.append(str(source["instance_id"]))
                    return {"instance_id": source["instance_id"]}

            def fake_judge(**kwargs):
                task_id = str(kwargs["instance"]["instance_id"])
                return (
                    InstanceResult(
                        task_id,
                        RunArtifacts(
                            instance_id=task_id,
                            run_dir=judge_run_dir,
                            trajectory_path=judge_out / "trajectory.jsonl",
                            status_code=0,
                        ),
                        None,
                        attempts=1,
                    ),
                    ["judge-base"],
                    1,
                    ["gsb_judged"],
                )

            args = argparse.Namespace(
                run_id="solver",
                judge_concurrency=1,
                concurrency=1,
                judge_semantic_max_attempts=1,
                judge_mode="gsb",
            )
            phase = run_gdpval._StreamingJudgePhase(
                args=args,
                suite=FakeSuite(),
                source_by_id={"ready": {"instance_id": "ready"}},
                run_root=root,
                base_judge_run_id="judge-base",
                wheelhouse=None,
                judge_provider="fake",
                judge_api_kind="openai-chat",
                provider_env={},
            )

            with (
                mock.patch.object(
                    run_gdpval,
                    "_run_streaming_judge_instance_with_semantic_retries",
                    side_effect=fake_judge,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                phase.submit_solver_result(solver_result)
                self.assertEqual(started, ["ready"])
                self.assertEqual(len(phase.futures), 1)
                summary = phase.finish()

            self.assertIsNotNone(summary)
            self.assertEqual(summary["judged"], 1)
            self.assertEqual(summary["mean_score"], 1.0)
            payload = json.loads((root / "solver" / "judge_summary.json").read_text())
            self.assertEqual(payload["rows"][0]["task_id"], "ready")


if __name__ == "__main__":
    unittest.main()
