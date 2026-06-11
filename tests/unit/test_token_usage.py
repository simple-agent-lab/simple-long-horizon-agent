"""Tests for token-usage propagation and context-view's use of it."""

from __future__ import annotations

import math
import unittest

from simple_agent_lab import (
    AssistantMessage,
    TokenUsage,
    assistant_message,
    build_context_view,
    effective_token_budget,
    estimate_context_tokens,
    estimate_message_chars,
    estimate_message_tokens,
    runtime_message,
    user_message,
)
from simple_agent_lab import (
    State,
    make_llm_agent,
    run,
)
from simple_agent_lab.context_view import CHARS_PER_TOKEN
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm.bridge import (
    _usage_or_none,
    llm_response_to_assistant_message,
)
from simple_agent_lab.llm.types import LLMResponse, TextBlock
from simple_agent_lab.messages import tool_result_message


class TokenUsageDataclassTest(unittest.TestCase):
    def test_defaults_are_zero(self) -> None:
        usage = TokenUsage()
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.cache_read_tokens, 0)
        self.assertEqual(usage.cache_write_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)

    def test_total_tokens_excludes_cache_to_match_billing_intent(self) -> None:
        # `total_tokens` is intentionally input + output only. Cache tokens
        # are reported separately because they price differently and
        # double-counting them in a single "total" misleads cost reporting.
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=500,
            cache_write_tokens=300,
        )
        self.assertEqual(usage.total_tokens, 120)

    def test_is_frozen_and_hashable(self) -> None:
        usage = TokenUsage(input_tokens=1, output_tokens=2)
        with self.assertRaises(Exception):
            usage.input_tokens = 99  # type: ignore[misc]
        # Hashing means it can live inside frozen AssistantMessage equality.
        self.assertEqual(hash(usage), hash(TokenUsage(input_tokens=1, output_tokens=2)))


class FromInclusiveInputTest(unittest.TestCase):
    """`from_inclusive_input` converts subset-cache reporting to additive.

    This is the one place the "cache is a subset of the input total" shape
    (OpenAI) becomes the project's "cache is additive to input_tokens" shape,
    so every OpenAI-family adapter shares it instead of re-deriving it.
    """

    def test_subtracts_cached_so_context_tokens_is_true_window(self) -> None:
        usage = TokenUsage.from_inclusive_input(
            total_input=2007, output=90, cached_read=1984
        )
        self.assertEqual(usage.input_tokens, 23)
        self.assertEqual(usage.cache_read_tokens, 1984)
        # 23 + 90 + 1984 — the real window, not the double-counted 4081.
        self.assertEqual(usage.context_tokens, 2097)

    def test_subtracts_both_cache_read_and_write(self) -> None:
        usage = TokenUsage.from_inclusive_input(
            total_input=100, output=10, cached_read=30, cache_write=20
        )
        self.assertEqual(usage.input_tokens, 50)  # 100 - 30 - 20
        self.assertEqual(usage.context_tokens, 110)  # back to total_input + output

    def test_no_cache_leaves_input_untouched(self) -> None:
        usage = TokenUsage.from_inclusive_input(total_input=42, output=7)
        self.assertEqual(usage.input_tokens, 42)
        self.assertEqual(usage.cache_read_tokens, 0)

    def test_floors_at_zero_if_cache_exceeds_total(self) -> None:
        # Defensive: a malformed provider reading must never yield negatives.
        usage = TokenUsage.from_inclusive_input(
            total_input=10, output=1, cached_read=99
        )
        self.assertEqual(usage.input_tokens, 0)


class AssistantMessageUsageTest(unittest.TestCase):
    def test_default_usage_is_none(self) -> None:
        message = assistant_message("hi")
        self.assertIsNone(message.usage)

    def test_usage_propagates_through_factory(self) -> None:
        usage = TokenUsage(input_tokens=42, output_tokens=7)
        message = assistant_message("hi", usage=usage)
        self.assertEqual(message.usage, usage)

    def test_messages_with_same_usage_compare_equal(self) -> None:
        a = assistant_message("hi", usage=TokenUsage(output_tokens=5))
        b = assistant_message("hi", usage=TokenUsage(output_tokens=5))
        self.assertEqual(a, b)

    def test_messages_with_different_usage_compare_unequal(self) -> None:
        a = assistant_message("hi", usage=TokenUsage(output_tokens=5))
        b = assistant_message("hi", usage=TokenUsage(output_tokens=6))
        self.assertNotEqual(a, b)


class TranslateUsageTest(unittest.TestCase):
    """`_usage_or_none` is the bridge from provider Usage → message-side TokenUsage."""

    def test_non_zero_translates_field_for_field(self) -> None:
        translated = _usage_or_none(
            TokenUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=30,
                cache_write_tokens=40,
            )
        )
        self.assertEqual(
            translated,
            TokenUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=30,
                cache_write_tokens=40,
            ),
        )

    def test_all_zero_returns_none_to_signal_unknown(self) -> None:
        # The default Usage() is all zeros — that is the "we have no data"
        # state, not "the call cost zero tokens". Translating it to a real
        # TokenUsage(0, 0, ...) would let downstream code mistake it for an
        # authoritative reading.
        self.assertIsNone(_usage_or_none(TokenUsage()))

    def test_only_cache_field_set_still_translates(self) -> None:
        # Cache-only is a real (if rare) state — e.g. a re-issued request that
        # hits a perfect cache. Don't drop it just because input/output are 0.
        translated = _usage_or_none(TokenUsage(cache_read_tokens=99))
        self.assertIsNotNone(translated)
        assert translated is not None  # narrow for type-checker
        self.assertEqual(translated.cache_read_tokens, 99)


class LLMResponseToAssistantMessageTest(unittest.TestCase):
    def test_usage_rides_into_runtime_message(self) -> None:
        response = LLMResponse(
            content=[TextBlock(text="ok")],
            usage=TokenUsage(input_tokens=15, output_tokens=3),
        )
        message = llm_response_to_assistant_message(
            response,
            sender="agent",
            target="user",
            kind="final",
        )
        assert isinstance(message, AssistantMessage)
        self.assertIsNotNone(message.usage)
        assert message.usage is not None
        self.assertEqual(message.usage.input_tokens, 15)
        self.assertEqual(message.usage.output_tokens, 3)

    def test_zero_usage_response_yields_no_usage_on_message(self) -> None:
        # Some adapters (or replay paths) emit responses with no usage data.
        # The runtime message should reflect "unknown" rather than fabricate
        # zeros that look authoritative.
        response = LLMResponse(content=[TextBlock(text="ok")])
        message = llm_response_to_assistant_message(
            response,
            sender="agent",
            target="user",
            kind="final",
        )
        assert isinstance(message, AssistantMessage)
        self.assertIsNone(message.usage)


class EstimateMessageTokensTest(unittest.TestCase):
    def test_assistant_with_output_tokens_uses_exact_count(self) -> None:
        message = assistant_message(
            "x" * 1000,  # large char-fallback estimate
            usage=TokenUsage(input_tokens=999_999, output_tokens=42),
        )
        # The exact `output_tokens=42` wins over the 250 char-estimate.
        self.assertEqual(estimate_message_tokens(message), 42)

    def test_assistant_with_zero_output_tokens_falls_back_to_chars(self) -> None:
        # output_tokens=0 means "we asked and the answer was zero" but in
        # practice, no real assistant message has zero output tokens — we
        # treat it as "unusable, fall back to estimation". This also keeps
        # the char fallback the safety net when usage data is partial or fabricated.
        message = assistant_message(
            "hello",
            usage=TokenUsage(input_tokens=500, output_tokens=0),
        )
        expected = math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_message_tokens(message), expected)

    def test_assistant_without_usage_falls_back_to_chars(self) -> None:
        message = assistant_message("hello world")
        expected = math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_message_tokens(message), expected)

    def test_user_message_always_falls_back_even_if_long(self) -> None:
        # No provider reports per-message tokens for user inputs, so user
        # messages always go through char-estimation.
        message = user_message("x" * 100)
        expected = math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_message_tokens(message), expected)

    def test_runtime_message_falls_back(self) -> None:
        message = runtime_message("be helpful")
        expected = math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_message_tokens(message), expected)

    def test_tool_result_falls_back(self) -> None:
        message = tool_result_message(
            "found 3 files",
            tool_call_id="call_1",
            tool_name="bash",
            target="agent",
        )
        expected = math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_message_tokens(message), expected)

    def test_fallback_rounds_up_via_ceiling(self) -> None:
        # The char fallback must round UP, not down — otherwise a 1-char
        # message reads as 0 tokens and a budgeting caller could treat it
        # as "free".
        message = user_message("x")
        self.assertGreaterEqual(estimate_message_tokens(message), 1)

    def test_default_chars_per_token_is_below_the_old_guess(self) -> None:
        # The default is a rounded neutral ratio below the old 4.0 guess, so the
        # char fallback no longer under-counts dense content as badly.
        self.assertLess(CHARS_PER_TOKEN, 4)
        message = user_message("x" * 400, target="agent")
        chars = estimate_message_chars(message)
        self.assertEqual(
            estimate_message_tokens(message), math.ceil(chars / CHARS_PER_TOKEN)
        )


class ContextStatsUsesUsageTest(unittest.TestCase):
    def test_estimated_tokens_prefers_known_usage(self) -> None:
        big_text = "x" * 1000  # would estimate to ~250 tokens via chars
        messages = [
            user_message("hi", target="agent"),
            assistant_message(
                big_text,
                sender="agent",
                target="user",
                kind="final",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ]
        view = build_context_view("agent", messages)

        # The assistant contributes its exact 5 output_tokens. The user
        # message still falls back. Total tokens must be lower than what a
        # pure char-based estimate would produce, since output_tokens=5 is tiny.
        char_only_estimate = math.ceil(view.stats.estimated_chars / CHARS_PER_TOKEN)
        self.assertLess(view.stats.estimated_tokens, char_only_estimate)
        self.assertEqual(view.stats.usage_known_messages, 1)

    def test_context_tokens_use_latest_usage_plus_trailing_estimates(self) -> None:
        u1 = user_message("older prompt", target="agent")
        a1 = assistant_message(
            "first answer",
            sender="agent",
            target="user",
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=5,
                cache_write_tokens=7,
            ),
        )
        u2 = user_message("new tool result or follow-up", target="agent")

        expected = 132 + estimate_message_tokens(u2)

        self.assertEqual(estimate_context_tokens([u1, a1, u2]), expected)
        self.assertEqual(
            build_context_view("agent", [u1, a1, u2]).stats.estimated_tokens, expected
        )

    def test_no_usage_anywhere_matches_char_quotient(self) -> None:
        messages = [
            user_message("hi there", target="agent"),
            assistant_message("hello back", sender="agent", target="user"),
        ]
        view = build_context_view("agent", messages)
        # No usage anywhere -> pure char fallback, summed per message (each
        # message rounds up independently, so this is sum-of-ceilings, not a
        # single ceiling over the combined character count).
        char_only = sum(
            math.ceil(estimate_message_chars(message) / CHARS_PER_TOKEN)
            for message in messages
        )
        self.assertEqual(view.stats.estimated_tokens, char_only)
        self.assertEqual(view.stats.usage_known_messages, 0)

    def test_estimated_tokens_use_latest_usage_baseline_when_mixed(self) -> None:
        # The latest usage-bearing assistant is the authoritative prefix size.
        # Messages after it are still estimated one by one.
        u = user_message("ask", target="agent")
        a1 = assistant_message(
            "first answer",
            sender="agent",
            target="user",
            usage=TokenUsage(input_tokens=20, output_tokens=11),
        )
        a2 = assistant_message("second answer", sender="agent", target="user")
        view = build_context_view("agent", [u, a1, a2])

        expected = 31 + estimate_message_tokens(a2)
        self.assertEqual(view.stats.estimated_tokens, expected)
        # Sanity: a1's contribution is the exact 11.
        self.assertEqual(estimate_message_tokens(a1), 11)
        self.assertEqual(view.stats.usage_known_messages, 1)

    def test_usage_known_messages_in_stats_dict(self) -> None:
        # Surface in as_dict() so the runtime trace records it.
        message = assistant_message(
            "hi",
            sender="agent",
            target="user",
            usage=TokenUsage(output_tokens=4),
        )
        view = build_context_view("agent", [message])
        self.assertIn("usage_known_messages", view.stats.as_dict())
        self.assertEqual(view.stats.as_dict()["usage_known_messages"], 1)


class EffectiveTokenBudgetTest(unittest.TestCase):
    def test_reserves_output_and_buffer_from_the_window(self) -> None:
        self.assertEqual(
            effective_token_budget(
                200_000, output_reserve=32_000, safety_buffer=20_000
            ),
            148_000,
        )

    def test_floors_at_zero_for_small_windows(self) -> None:
        self.assertEqual(
            effective_token_budget(8_000, output_reserve=32_000, safety_buffer=20_000),
            0,
        )

    def test_larger_buffer_yields_smaller_budget(self) -> None:
        loose = effective_token_budget(100_000, safety_buffer=10_000)
        tight = effective_token_budget(100_000, safety_buffer=40_000)
        self.assertGreater(loose, tight)


class EndToEndUsagePropagationTest(unittest.TestCase):
    """End-to-end check: a real model turn must land usage on the runtime message.

    This covers the wiring between make_llm_agent → fake adapter → bridge
    → AssistantMessage. If any link drops the field, all the unit tests above
    can still pass while real runs lose usage data.
    """

    def test_make_llm_agent_records_usage_on_assistant_message(self) -> None:
        provider = LLMProvider(id="fake", api="fake", model="fake-model")
        agent = make_llm_agent(
            name="writer", provider=provider, role="say hi", target="user"
        )
        state = State("say hi please")
        state.send("task", "user", "writer", state.task)

        for _ in run(agent, state):
            pass

        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        assert isinstance(final, AssistantMessage)
        self.assertIsNotNone(
            final.usage, "fake adapter always reports usage; runtime must keep it"
        )
        assert final.usage is not None
        # The fake adapter sets `output_tokens = max(1, len(text)//4)` and
        # `input_tokens = sum(estimate per message)` — both must be > 0.
        self.assertGreater(final.usage.output_tokens, 0)
        self.assertGreater(final.usage.input_tokens, 0)


if __name__ == "__main__":
    unittest.main()
