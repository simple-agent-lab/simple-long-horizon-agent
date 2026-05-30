"""Offline smoke for the compression eval suite.

Mirrors the SWE-bench adapter smoke tests: exercises the model-free half of
`evals/compression` so CI guards the scorers and scenarios without a model.
The live half (real token counts, real summaries) is not covered here.
"""

from __future__ import annotations

import unittest

from simple_agent_lab.compression import ToolCompactStrategy
from simple_agent_lab.messages import (
    ToolCallBlock,
    assistant_message,
    tool_result_message,
)
from simple_agent_lab.state import State

from evals.compression.metrics import (
    fact_recall,
    pinned_kinds_present,
    run_compression,
    tool_pairs_intact,
)
from evals.compression.run_eval import run_offline
from evals.compression.scenarios import ALL_SCENARIOS, TOOL_HEAVY


class ScenarioTest(unittest.TestCase):
    def test_every_needle_is_literally_present_in_its_source(self) -> None:
        # fact_recall scores by substring, so each needle must actually appear
        # in the planted text — otherwise the metric can never see a hit.
        for scenario in ALL_SCENARIOS:
            for fact in scenario.facts:
                self.assertIn(
                    fact.needle,
                    fact.text,
                    f"{scenario.name}: needle {fact.needle!r} missing from source",
                )

    def test_scenarios_build_fresh_independent_state(self) -> None:
        for scenario in ALL_SCENARIOS:
            first = scenario.build()
            second = scenario.build()
            self.assertIsNot(first, second)
            self.assertGreater(len(first.messages), scenario.keep_recent)


class ScorerTest(unittest.TestCase):
    def test_tool_pairs_intact_detects_orphans(self) -> None:
        call = assistant_message(
            [ToolCallBlock("c0", "echo", {})],
            sender="w",
            target="u",
        )
        result = tool_result_message(
            "ok", tool_call_id="c0", tool_name="echo", target="w"
        )
        self.assertTrue(tool_pairs_intact([call, result]))
        # Drop the result -> the call is now an orphan.
        self.assertFalse(tool_pairs_intact([call]))

    def test_pinned_kinds_present_flags_dropped_pin(self) -> None:
        state = State("t")
        task = state.send("task", "user", "w", "the task")
        note = state.send("message", "user", "w", "a note")
        self.assertTrue(pinned_kinds_present([task, note], [task], ("task",)))
        self.assertFalse(pinned_kinds_present([task, note], [note], ("task",)))

    def test_fact_recall_counts_substring_hits(self) -> None:
        hits, rate = fact_recall("kept alpha only", TOOL_HEAVY.facts[:1])
        self.assertEqual(rate, 0.0)  # needle not in text
        hits, rate = fact_recall("disk-93pct full", (TOOL_HEAVY.facts[2],))
        self.assertEqual(rate, 1.0)


class CompactPathTest(unittest.TestCase):
    def test_tool_compact_keeps_findings_in_preview(self) -> None:
        # The rule-based strategy keeps a 200-char result preview, so the
        # distinctive findings should survive verbatim in the compact marker.
        scenario = TOOL_HEAVY
        outcome = run_compression(
            scenario.build(),
            ToolCompactStrategy(threshold_tokens=1, keep_recent_exchanges=1),
        )
        self.assertTrue(outcome.triggered)
        self.assertLess(outcome.ratio, 1.0)
        self.assertTrue(tool_pairs_intact(outcome.active_messages))
        hits, rate = fact_recall(outcome.summary_text, scenario.facts[:2])
        self.assertEqual(rate, 1.0)


class OfflineSuiteTest(unittest.TestCase):
    def test_run_offline_all_checks_pass(self) -> None:
        results = run_offline(ALL_SCENARIOS)
        self.assertTrue(results)
        for result in results:
            self.assertTrue(result.passed, f"{result.trace_id}: {result.reason}")
        scorers = {result.scorer for result in results}
        self.assertIn("threshold_trigger_curve", scorers)
        self.assertIn("compression_pass", scorers)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
