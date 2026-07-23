from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from simple_agent_lab.evals.suites.onemillion import container
from simple_agent_lab.llm import Provider

RUBRICS = [
    {"rubric_number": 1, "rubric_detail": "Defines the concept.", "rubric_weight": 10},
    {"rubric_number": 2, "rubric_detail": "Gives an example.", "rubric_weight": 5},
]


class OneMillionContainerTest(unittest.TestCase):
    def test_build_task_includes_system_prompt_and_question(self) -> None:
        task = container.build_task(
            {"system_prompt": "Be precise.", "prompt": "What is X?"},
            workdir="/workspace",
        )
        self.assertIn("Be precise.", task)
        self.assertIn("What is X?", task)

    def test_extract_result_reads_written_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / container.RESPONSE_FILENAME).write_text(
                "the answer", encoding="utf-8"
            )
            result = container.extract_result(
                Path(tmp), {"case_id": 7, "instance_id": "case_7"}
            )
        self.assertEqual(result["model_response"], "the answer")
        self.assertEqual(result["case_id"], 7)
        self.assertEqual(result["instance_id"], "case_7")

    def test_build_agent_persists_answer_to_workspace(self) -> None:
        fake = Provider(id="fake", api="fake", model="fake-model")
        with tempfile.TemporaryDirectory() as tmp:
            agent = container.build_agent(provider=fake, cwd=Path(tmp))
            _state, events = agent.run("Answer the question.", max_turns=1)
            list(events)  # drive the loop to completion
            written = (Path(tmp) / container.RESPONSE_FILENAME).read_text(
                encoding="utf-8"
            )
        self.assertTrue(written.strip())

    def test_evaluate_grades_response_against_staged_rubrics(self) -> None:
        judge_json = """```json
[
  {"rubric_id": 1, "status": "是", "justification": "ok"},
  {"rubric_id": 2, "status": "否", "justification": "no example"}
]
```"""
        env = {"JUDGE_MODEL": "judge-x", "JUDGE_AUTH_TOKEN": "secret"}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / container.RESPONSE_FILENAME).write_text(
                "my answer", encoding="utf-8"
            )
            context = {"eval": {"prompt": "Q?", "rubrics": RUBRICS, "human_scores": {}}}
            with (
                mock.patch.dict("os.environ", env, clear=False),
                mock.patch.object(
                    container,
                    "complete_with_retry",
                    return_value=SimpleNamespace(text=judge_json),
                ),
            ):
                verdict = container.evaluate(
                    Path(tmp), {"prompt": "Q?"}, context=context
                )

        self.assertTrue(verdict["scored"])
        self.assertEqual(verdict["judge_model"], "judge-x")
        self.assertEqual(verdict["total_score"], 10)  # rubric 1 hit (+10), 2 miss
        self.assertEqual(verdict["max_score"], 15)
        self.assertAlmostEqual(verdict["score"], 10 / 15)
        self.assertEqual(verdict["rubric_scores"]["1"], 10)
        self.assertEqual(verdict["rubric_scores"]["2"], 0)

    def test_evaluate_without_rubrics_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict = container.evaluate(Path(tmp), {}, context={"eval": {}})
        self.assertFalse(verdict["scored"])
        self.assertEqual(verdict["status"], "no_rubrics")

    def test_judge_provider_falls_back_to_openai_env(self) -> None:
        env = {
            "OPENAI_MODEL": "gen-model",
            "OPENAI_AUTH_TOKEN": "gen-token",
            "OPENAI_BASE_URL": "https://example/v1",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            provider = container.judge_provider_from_env()
        self.assertEqual(provider.model, "gen-model")
        self.assertEqual(provider.api_key_env, "OPENAI_AUTH_TOKEN")
        self.assertEqual(provider.base_url, "https://example/v1")
