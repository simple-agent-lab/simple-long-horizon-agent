"""Tests for the model-metadata layer: TokenUsage -> dollars, plus context windows."""

from __future__ import annotations

import json
import unittest

from simple_agent_lab import (
    ContextWindowBook,
    CostBreakdown,
    ModelPrice,
    PriceBook,
    RunCost,
    TokenUsage,
    default_price_book,
    usage_cost,
)
from simple_agent_lab.model_metadata import (
    CACHE_READ_RATIO,
    CACHE_WRITE_RATIO,
    CONTEXT_WINDOW_BOOK_ENV,
    DEFAULT_PRICES,
    PRICE_BOOK_ENV,
    default_context_window_book,
)
from simple_agent_lab.trace.run_trace import RunTrace


class ModelPriceTest(unittest.TestCase):
    def test_from_mapping_defaults_cache_to_standard_multiples(self) -> None:
        # cache read AND write are priced off the input rate, each with its own
        # multiplier, never folded in. An override that gives only input/output
        # gets the standard cache multiples filled in.
        price = ModelPrice.from_mapping({"input": 1.0, "output": 5.0})
        self.assertEqual(price.cache_read, round(1.0 * CACHE_READ_RATIO, 6))
        self.assertEqual(price.cache_write, round(1.0 * CACHE_WRITE_RATIO, 6))
        self.assertEqual(price.cache_read, 0.1)
        self.assertEqual(price.cache_write, 1.25)

    def test_from_mapping_respects_explicit_cache_rates(self) -> None:
        price = ModelPrice.from_mapping(
            {"input": 1.0, "output": 5.0, "cache_read": 0.05, "cache_write": 2.0}
        )
        self.assertEqual(price.cache_read, 0.05)
        self.assertEqual(price.cache_write, 2.0)


class UsageCostTest(unittest.TestCase):
    def test_each_bucket_priced_independently(self) -> None:
        price = ModelPrice(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
        )
        cost = usage_cost(usage, price)
        self.assertEqual(cost.input_usd, 5.0)
        self.assertEqual(cost.output_usd, 25.0)
        self.assertEqual(cost.cache_read_usd, 0.5)
        self.assertEqual(cost.cache_write_usd, 6.25)
        self.assertEqual(cost.total_usd, 36.75)

    def test_partial_million_scales_linearly(self) -> None:
        price = ModelPrice(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
        cost = usage_cost(TokenUsage(input_tokens=200, output_tokens=88), price)
        self.assertAlmostEqual(cost.input_usd, 200 * 5.0 / 1_000_000)
        self.assertAlmostEqual(cost.output_usd, 88 * 25.0 / 1_000_000)

    def test_breakdown_adds(self) -> None:
        a = CostBreakdown(input_usd=1.0, output_usd=2.0)
        b = CostBreakdown(input_usd=0.5, cache_read_usd=0.25)
        total = a + b
        self.assertEqual(total.input_usd, 1.5)
        self.assertEqual(total.output_usd, 2.0)
        self.assertEqual(total.cache_read_usd, 0.25)
        self.assertEqual(total.total_usd, 3.75)


class PriceBookLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.book = PriceBook(DEFAULT_PRICES)

    def test_exact_alias(self) -> None:
        self.assertIsNotNone(self.book.price_for("claude-opus-4-8"))

    def test_dated_snapshot_resolves_to_alias(self) -> None:
        # Adapters resolve aliases to dated snapshots on the wire; the rate
        # card only lists bare aliases, so the dated id must still match.
        price = self.book.price_for("claude-opus-4-8-20260101")
        self.assertEqual(price, DEFAULT_PRICES["claude-opus-4-8"])

    def test_provider_prefix_stripped(self) -> None:
        self.assertEqual(
            self.book.price_for("anthropic/claude-sonnet-4-6"),
            DEFAULT_PRICES["claude-sonnet-4-6"],
        )

    def test_bedrock_style_dotted_prefix(self) -> None:
        self.assertEqual(
            self.book.price_for("anthropic.claude-haiku-4-5"),
            DEFAULT_PRICES["claude-haiku-4-5"],
        )

    def test_unknown_model_returns_none(self) -> None:
        self.assertIsNone(self.book.price_for("no-such-model-xyz"))

    def test_empty_model_returns_none(self) -> None:
        self.assertIsNone(self.book.price_for(""))


class ContextWindowBookLookupTest(unittest.TestCase):
    def test_default_context_windows_include_models_dev_values(self) -> None:
        book = default_context_window_book()

        self.assertEqual(book.window_for("deepseek-v4-flash"), 1_000_000)
        self.assertEqual(book.window_for("anthropic/claude-sonnet-4-6"), 1_000_000)
        self.assertEqual(book.window_for("z-ai/glm-5.2"), 1_000_000)
        self.assertEqual(book.window_for("zhipuai/glm-5.2"), 1_000_000)
        self.assertEqual(book.window_for("gpt-5.4"), 1_000_000)
        self.assertEqual(book.window_for("gpt-5.5"), 1_000_000)
        self.assertEqual(book.window_for("gpt-5.3-codex"), 1_000_000)
        # The platform deployment id must resolve via substring match — this is
        # exactly the id shape that previously missed and fell back to the fixed
        # compression-threshold default.
        self.assertEqual(
            book.window_for("deployment-gpt-5.4-2026-03-05-platform-global"),
            1_000_000,
        )

    def test_dated_snapshot_resolves_to_alias(self) -> None:
        book = ContextWindowBook({"claude-sonnet-4-6": 200_000})

        self.assertEqual(
            book.window_for("anthropic/claude-sonnet-4-6-20260101"),
            200_000,
        )

    def test_provider_prefixed_key_resolves_to_bare_alias(self) -> None:
        book = ContextWindowBook({"anthropic.claude-sonnet-4-6": 200_000})

        self.assertEqual(book.window_for("claude-sonnet-4-6"), 200_000)

    def test_dotted_version_key_does_not_match_unrelated_ids(self) -> None:
        # Regression: a dotted-version key ("glm-5.2") must not produce a bare
        # "2" alias that substring-matches any unrelated model id containing a
        # "2". Such an id should miss entirely, not borrow glm-5.2's window.
        book = ContextWindowBook({"glm-5.2": 1_000_000})

        self.assertEqual(book.window_for("glm-5.2"), 1_000_000)
        self.assertIsNone(book.window_for("totally-unknown-2-model"))
        self.assertIsNone(book.window_for("claude-opus-4-2-20260101"))

    def test_litellm_env_file_adds_models(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "windows.json"
            path.write_text(
                json.dumps(
                    {
                        "sample_spec": {"max_input_tokens": "documentation"},
                        "test-provider.my-model": {
                            "litellm_provider": "test",
                            "max_input_tokens": 123_456,
                        },
                    }
                ),
                encoding="utf-8",
            )

            old = os.environ.get(CONTEXT_WINDOW_BOOK_ENV)
            os.environ[CONTEXT_WINDOW_BOOK_ENV] = str(path)
            try:
                book = default_context_window_book()
            finally:
                if old is None:
                    del os.environ[CONTEXT_WINDOW_BOOK_ENV]
                else:
                    os.environ[CONTEXT_WINDOW_BOOK_ENV] = old

        self.assertEqual(book.window_for("my-model"), 123_456)

    def test_models_dev_models_json_adds_context_windows(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "deepseek/deepseek-v4-flash": {
                            "name": "DeepSeek V4 Flash",
                            "limit": {
                                "context": 1_000_000,
                                "output": 384_000,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            old = os.environ.get(CONTEXT_WINDOW_BOOK_ENV)
            os.environ[CONTEXT_WINDOW_BOOK_ENV] = str(path)
            try:
                book = default_context_window_book()
            finally:
                if old is None:
                    del os.environ[CONTEXT_WINDOW_BOOK_ENV]
                else:
                    os.environ[CONTEXT_WINDOW_BOOK_ENV] = old

        self.assertEqual(book.window_for("deepseek-v4-flash"), 1_000_000)

    def test_models_dev_api_json_provider_models_add_context_windows(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api.json"
            path.write_text(
                json.dumps(
                    {
                        "openai": {
                            "id": "openai",
                            "models": {
                                "gpt-5": {
                                    "id": "gpt-5",
                                    "limit": {
                                        "context": 400_000,
                                        "output": 128_000,
                                    },
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            old = os.environ.get(CONTEXT_WINDOW_BOOK_ENV)
            os.environ[CONTEXT_WINDOW_BOOK_ENV] = str(path)
            try:
                book = default_context_window_book()
            finally:
                if old is None:
                    del os.environ[CONTEXT_WINDOW_BOOK_ENV]
                else:
                    os.environ[CONTEXT_WINDOW_BOOK_ENV] = old

        self.assertEqual(book.window_for("gpt-5"), 400_000)
        self.assertEqual(book.window_for("openai/gpt-5"), 400_000)


class EnvOverrideTest(unittest.TestCase):
    def test_env_file_overrides_and_adds_models(
        self,
    ) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.json"
            path.write_text(
                json.dumps(
                    {
                        "claude-opus-4-8": {"input": 99.0, "output": 1.0},
                        "my-model": {"input": 2.0, "output": 4.0},
                    }
                ),
                encoding="utf-8",
            )
            import os

            old = os.environ.get(PRICE_BOOK_ENV)
            os.environ[PRICE_BOOK_ENV] = str(path)
            try:
                book = default_price_book()
            finally:
                if old is None:
                    del os.environ[PRICE_BOOK_ENV]
                else:
                    os.environ[PRICE_BOOK_ENV] = old

            opus = book.price_for("claude-opus-4-8")
            assert opus is not None
            self.assertEqual(opus.input, 99.0)  # overridden
            custom = book.price_for("my-model")
            assert custom is not None
            self.assertEqual(custom.input, 2.0)  # added


class RunCostFromCallsTest(unittest.TestCase):
    def test_groups_and_sorts_by_cost(self) -> None:
        calls = [
            ("claude-haiku-4-5", TokenUsage(input_tokens=1000, output_tokens=10)),
            ("claude-opus-4-8", TokenUsage(input_tokens=1000, output_tokens=1000)),
            ("claude-opus-4-8", TokenUsage(input_tokens=500, output_tokens=0)),
        ]
        run_cost = RunCost.from_calls(calls, PriceBook(DEFAULT_PRICES))
        self.assertEqual(len(run_cost.by_model), 2)
        # Opus is the most expensive -> sorts first.
        self.assertEqual(run_cost.by_model[0].model, "claude-opus-4-8")
        self.assertEqual(run_cost.by_model[0].calls, 2)
        self.assertEqual(run_cost.calls, 3)
        # Opus tokens summed across its two calls.
        self.assertEqual(run_cost.by_model[0].tokens.input_tokens, 1500)

    def test_unpriced_model_counted_but_zero_dollars(self) -> None:
        calls = [
            ("no-such-model-xyz", TokenUsage(input_tokens=1000, output_tokens=1000))
        ]
        run_cost = RunCost.from_calls(calls, PriceBook(DEFAULT_PRICES))
        self.assertEqual(run_cost.total_usd, 0.0)
        self.assertEqual(run_cost.unpriced_models, ("no-such-model-xyz",))
        self.assertEqual(run_cost.by_model[0].tokens.output_tokens, 1000)


class RunCostFromRunTest(unittest.TestCase):
    """`from_run` reads model_response events + descends into sub-agent logs."""

    def test_dict_events_main_agent(self) -> None:
        events = [
            {
                "kind": "model_response",
                "model": "claude-opus-4-8",
                "usage": {"input_tokens": 1000, "output_tokens": 1000},
            },
            {"kind": "turn_end"},
        ]
        run_cost = RunCost.from_run(events, [], PriceBook(DEFAULT_PRICES))
        self.assertEqual(run_cost.calls, 1)
        self.assertAlmostEqual(
            run_cost.total_usd, (1000 * 5.0 + 1000 * 25.0) / 1_000_000
        )

    def test_zero_usage_event_skipped(self) -> None:
        events = [
            {
                "kind": "model_response",
                "model": "claude-opus-4-8",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        ]
        run_cost = RunCost.from_run(events, [], PriceBook(DEFAULT_PRICES))
        self.assertEqual(run_cost.calls, 0)
        self.assertEqual(run_cost.by_model, ())

    def test_sub_agent_calls_included(self) -> None:
        # A tool-result message carries the sub-agent's event log under
        # sidecar.details[call_id].sub_events, exactly as task_tool records it.
        main_events = [
            {
                "kind": "model_response",
                "model": "claude-opus-4-8",
                "usage": {"input_tokens": 1000, "output_tokens": 0},
            },
        ]
        messages = [
            {
                "kind": "tool_result",
                "sidecar": {
                    "details": {
                        "call_1": {
                            "sub_events": [
                                {
                                    "kind": "model_response",
                                    "model": "claude-haiku-4-5",
                                    "usage": {
                                        "input_tokens": 2000,
                                        "output_tokens": 0,
                                    },
                                }
                            ]
                        }
                    }
                },
            }
        ]
        run_cost = RunCost.from_run(main_events, messages, PriceBook(DEFAULT_PRICES))
        models = {entry.model for entry in run_cost.by_model}
        self.assertEqual(models, {"claude-opus-4-8", "claude-haiku-4-5"})
        self.assertEqual(run_cost.calls, 2)


class TraceRecordEmbedsCostTest(unittest.TestCase):
    def test_cost_block_present_and_json_safe(self) -> None:
        trace = RunTrace(
            trace_id="t1",
            producer="test",
            task="do a thing",
            events=[
                {
                    "kind": "model_response",
                    "model": "claude-opus-4-8",
                    "usage": {"input_tokens": 1000, "output_tokens": 1000},
                }
            ],
            messages=[],
        )
        # Cost is derived from the event stream (not embedded in v5).
        cost = trace.run_cost().as_dict()
        self.assertGreaterEqual(cost["total_usd"], 0.0)
        self.assertEqual(cost["calls"], 1)
        # Round-trips through json without error -> genuinely JSON-safe.
        json.dumps(cost)


if __name__ == "__main__":
    unittest.main()
