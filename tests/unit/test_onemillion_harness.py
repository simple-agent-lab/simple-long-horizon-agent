from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.onemillion import harness


class OneMillionHarnessTest(unittest.TestCase):
    def test_sanitized_instance_drops_rubrics_and_grading(self) -> None:
        record = {
            "instance_id": "case_1",
            "prompt": "Q?",
            "rubrics": [{"rubric_number": 1}],
            "model_response": {"model_response_1": {}},
        }
        clean = harness.sanitized_instance(record)
        self.assertIn("prompt", clean)
        self.assertNotIn("rubrics", clean)
        self.assertNotIn("model_response", clean)

    def test_normalize_case_maps_capitalized_variants(self) -> None:
        record = harness.normalize_case(
            {"Prompt": "Q?", "Rubrics": [{"rubric_number": 1}], "System_Prompt": "S"}
        )
        self.assertEqual(record["prompt"], "Q?")
        self.assertEqual(record["rubrics"], [{"rubric_number": 1}])
        self.assertEqual(record["system_prompt"], "S")

    def test_instance_id_for_prefers_case_id(self) -> None:
        self.assertEqual(harness.instance_id_for({"case_id": 2860}), "case_2860")
        self.assertEqual(
            harness.instance_id_for({"instance_id": "given"}, fallback="x"), "given"
        )
        self.assertEqual(harness.instance_id_for({}, fallback="stem"), "stem")

    def test_eval_payload_returns_none_without_rubrics(self) -> None:
        self.assertIsNone(harness.eval_payload({"prompt": "Q?"}))
        payload = harness.eval_payload(
            {"prompt": "Q?", "rubrics": [{"rubric_number": 1}]}
        )
        assert payload is not None
        self.assertEqual(payload["rubrics"], [{"rubric_number": 1}])

    def test_load_dataset_reads_single_and_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single.json").write_text(
                json.dumps({"case_id": 11, "prompt": "Q1", "rubrics": []}),
                encoding="utf-8",
            )
            (root / "list.json").write_text(
                json.dumps(
                    [
                        {"case_id": 21, "prompt": "Q2", "rubrics": []},
                        {"case_id": 22, "prompt": "Q3", "rubrics": []},
                    ]
                ),
                encoding="utf-8",
            )
            records = harness.load_dataset(root)

        ids = {r["instance_id"] for r in records}
        self.assertEqual(ids, {"case_11", "case_21", "case_22"})

    def test_container_environment_requires_generator_creds(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                harness.container_environment("openai")

        env = {
            "OPENAI_MODEL": "m",
            "OPENAI_AUTH_TOKEN": "t",
            "JUDGE_MODEL": "j",
            "JUDGE_AUTH_TOKEN": "jt",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            resolved = harness.container_environment("openai")
        self.assertEqual(resolved["OPENAI_MODEL"], "m")
        self.assertEqual(resolved["JUDGE_MODEL"], "j")

    def test_container_environment_empty_for_fake_provider(self) -> None:
        self.assertEqual(harness.container_environment("fake"), {})
