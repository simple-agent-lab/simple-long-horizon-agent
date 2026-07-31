from __future__ import annotations

import unittest

from simple_long_horizon_agent.evals.suites.onemillion.grading import (
    build_grading_prompt,
    convert_scores,
    parse_grading_response,
    score_summary,
)

RUBRICS = [
    {"rubric_number": 1, "rubric_detail": "Defines viral titer.", "rubric_weight": 10},
    {"rubric_number": 2, "rubric_detail": "Lists 3 methods.", "rubric_weight": 8},
    {
        "rubric_number": 3,
        "rubric_detail": "Makes a factual error.",
        "rubric_weight": -5,
    },
]


class OneMillionGradingTest(unittest.TestCase):
    def test_build_grading_prompt_uses_no_human_template_and_lists_rubrics(
        self,
    ) -> None:
        prompt = build_grading_prompt("Q?", "A.", RUBRICS, human_scores_dict={})
        self.assertIn("Defines viral titer.", prompt)
        self.assertIn("rubricWeight: +10分", prompt)
        self.assertIn("rubricWeight: -5分", prompt)
        # The no-human template never asks for a consistency field.
        self.assertNotIn("humanScore", prompt)

    def test_parse_grading_response_reads_code_block_json(self) -> None:
        judge_text = """Here is my assessment:
```json
[
  {"rubric_id": 1, "status": "是", "justification": "ok"},
  {"rubric_id": 2, "status": "否", "justification": "missing"},
  {"rubric_id": 3, "status": "否", "justification": "no error"}
]
```
"""
        raw = parse_grading_response(judge_text, RUBRICS)
        self.assertEqual(raw[1]["binary_score"], 1)
        self.assertEqual(raw[2]["binary_score"], 0)
        self.assertEqual(raw[3]["binary_score"], 0)

    def test_parse_grading_response_fills_missing_rubric_as_na(self) -> None:
        judge_text = '[{"rubric_id": 1, "status": "是", "justification": "ok"}]'
        raw = parse_grading_response(judge_text, RUBRICS)
        self.assertEqual(raw[1]["binary_score"], 1)
        self.assertEqual(raw[2]["binary_score"], "NA")
        self.assertEqual(raw[3]["binary_score"], "NA")

    def test_convert_scores_weights_hits_for_positive_and_negative(self) -> None:
        raw = {
            1: {"binary_score": 1},  # +10 hit -> +10
            2: {"binary_score": 0},  # +8 miss -> 0
            3: {"binary_score": 1},  # -5 hit (penalty applies) -> -5
        }
        final = convert_scores(raw, RUBRICS)
        self.assertEqual(final[1], 10)
        self.assertEqual(final[2], 0)
        self.assertEqual(final[3], -5)

    def test_convert_scores_preserves_na(self) -> None:
        raw = {
            1: {"binary_score": "NA"},
            2: {"binary_score": 1},
            3: {"binary_score": 0},
        }
        final = convert_scores(raw, RUBRICS)
        self.assertEqual(final[1], "NA")
        self.assertEqual(final[2], 8)

    def test_score_summary_uses_positive_weight_total_for_accuracy(self) -> None:
        final = {1: 10, 2: 0, 3: 0}
        summary = score_summary(final, RUBRICS)
        self.assertEqual(summary["total_score"], 10)
        self.assertEqual(summary["max_score"], 18)
        self.assertEqual(summary["min_score"], -5)
        self.assertAlmostEqual(summary["accuracy"], 10 / 18)

    def test_score_summary_pure_penalty_case(self) -> None:
        rubrics = [{"rubric_number": 1, "rubric_detail": "x", "rubric_weight": -4}]
        # No penalty triggered -> perfect score for a pure-penalty case.
        self.assertAlmostEqual(score_summary({1: 0}, rubrics)["accuracy"], 1.0)
        # Penalty triggered -> accuracy degrades.
        self.assertAlmostEqual(score_summary({1: -4}, rubrics)["accuracy"], 0.0)
