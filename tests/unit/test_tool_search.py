from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.tool_search.synthetic import (
    build_synthetic_registry,
    run_bench,
    write_report,
)
from simple_agent_lab.tool_search import (
    BM25ToolRetriever,
    make_invoke_tool,
    make_search_tools_tool,
)
from simple_agent_lab.tools import tool_result_text


class ToolSearchCoreTest(unittest.TestCase):
    def test_bm25_retrieves_relevant_tool(self) -> None:
        registry = build_synthetic_registry(distractors=20)
        results = BM25ToolRetriever(registry).search("add two integers", k=3)

        self.assertTrue(results)
        self.assertEqual(results[0].record.tool.name, "add_numbers")

    def test_search_tool_returns_structured_candidates(self) -> None:
        registry = build_synthetic_registry(distractors=5)
        tool = make_search_tools_tool(registry)

        result = tool.execute(
            "s1",
            {"query": "reverse text characters", "k": 2},
            lambda: False,
            None,
        )

        self.assertFalse(result.is_error)
        payload = json.loads(tool_result_text(result))
        self.assertEqual(len(payload["tools"]), 2)
        self.assertIn("name", payload["tools"][0])
        self.assertIn("parameters", payload["tools"][0])

    def test_invoke_tool_executes_registered_tool(self) -> None:
        registry = build_synthetic_registry(distractors=0)
        invoke = make_invoke_tool(registry)

        result = invoke.execute(
            "i1",
            {"tool_name": "add_numbers", "arguments": {"a": 19, "b": 23}},
            lambda: False,
            None,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.details["tool_name"], "add_numbers")
        self.assertEqual(result.details["result_details"]["value"], 42)

    def test_invoke_tool_validates_schema(self) -> None:
        registry = build_synthetic_registry(distractors=0)
        invoke = make_invoke_tool(registry)

        result = invoke.execute(
            "i1",
            {"tool_name": "add_numbers", "arguments": {"a": 19}},
            lambda: False,
            None,
        )

        self.assertTrue(result.is_error)
        self.assertIn("Missing required argument", tool_result_text(result))


class SyntheticBenchTest(unittest.TestCase):
    def test_proxy_mode_searches_then_executes(self) -> None:
        registry = build_synthetic_registry(distractors=50)
        report = run_bench(registry=registry, mode="proxy", top_k=5)

        summary = report.summary()
        self.assertEqual(summary["runner"], "scripted")
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["correct_tool_rate"], 1.0)
        self.assertEqual(summary["gold_in_candidates_rate"], 1.0)
        self.assertGreater(summary["mean_gold_rank"], 0)
        self.assertGreater(summary["mean_peak_context_tokens"], 0)
        self.assertTrue(all(result.tool_calls == 2 for result in report.results))

    def test_llm_runner_requires_provider(self) -> None:
        registry = build_synthetic_registry(distractors=0)

        with self.assertRaisesRegex(ValueError, "requires a Provider"):
            run_bench(registry=registry, mode="proxy", runner="llm")

    def test_static_budgeted_can_fail_when_correct_tool_is_not_visible(self) -> None:
        registry = build_synthetic_registry(distractors=50)
        report = run_bench(
            registry=registry,
            mode="static_budgeted",
            static_tool_limit=1,
        )

        self.assertLess(report.summary()["success_rate"], 1.0)

    def test_static_budgeted_schema_cost_counts_visible_budget(self) -> None:
        registry = build_synthetic_registry(distractors=50)
        small = run_bench(
            registry=registry,
            mode="static_budgeted",
            static_tool_limit=2,
        )
        large = run_bench(
            registry=registry,
            mode="static_budgeted",
            static_tool_limit=20,
        )

        self.assertLess(
            small.summary()["mean_schema_tokens"],
            large.summary()["mean_schema_tokens"],
        )

    def test_write_report_persists_json(self) -> None:
        registry = build_synthetic_registry(distractors=5)
        report = run_bench(registry=registry, mode="dynamic_topk", top_k=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            write_report(path, [report])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["summaries"][0]["mode"], "dynamic_topk")
        self.assertEqual(payload["summaries"][0]["runner"], "scripted")
        self.assertIn("gold_rank", payload["results"][0])
        self.assertIn("gold_in_candidates", payload["results"][0])
        self.assertEqual(len(payload["results"]), len(report.results))


if __name__ == "__main__":
    unittest.main()
