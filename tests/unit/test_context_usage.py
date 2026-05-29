from __future__ import annotations

import unittest

from simple_agent_lab import (
    Agent,
    AssistantMessage,
    Message,
    ModelRequestEvent,
    State,
    TokenUsage,
    TurnEndEvent,
    assistant_message,
    estimate_message_tokens,
    run,
)


class ContextUsageTest(unittest.TestCase):
    def test_next_turn_uses_latest_usage_plus_trailing_message_estimate(self) -> None:
        # Hand-write a deterministic two-turn generator instead of going
        # through `make_llm_agent` + the fake adapter: this test cares about
        # a specific usage shape on turn 1, and threading that through a
        # provider would obscure what we're actually validating, which is
        # cross-turn context-token accounting (latest known usage + trailing
        # estimate).
        first_usage = TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=5,
            cache_write_tokens=7,
        )
        turn = {"n": 0}

        def two_turn_generate(visible: list[Message]) -> Message:
            del visible
            turn["n"] += 1
            if turn["n"] == 1:
                return assistant_message(
                    "first answer",
                    sender="writer",
                    target="user",
                    kind="thought",
                    usage=first_usage,
                )
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        agent = Agent("writer", two_turn_generate)
        state = State(
            "Explain why context accounting should use provider usage when available."
        )
        state.send("task", "user", "writer", state.task)

        trailing: Message | None = None
        for event in run(agent, state, max_turns=2):
            if isinstance(event, TurnEndEvent) and trailing is None:
                trailing = state.send(
                    "message",
                    "user",
                    "writer",
                    "This note arrived after the first model response.",
                )

        self.assertIsNotNone(trailing)
        assert trailing is not None

        requests = [
            event for event in state.events if isinstance(event, ModelRequestEvent)
        ]
        self.assertEqual(len(requests), 2)
        second_request = requests[1]
        self.assertEqual(second_request.visible_count, 3)

        assistants = [
            message
            for message in state.messages
            if isinstance(message, AssistantMessage) and message.sender == "writer"
        ]
        self.assertEqual(len(assistants), 2)
        self.assertEqual(assistants[0].usage, first_usage)

        usage_context_tokens = (
            first_usage.input_tokens
            + first_usage.output_tokens
            + first_usage.cache_read_tokens
            + first_usage.cache_write_tokens
        )
        expected = usage_context_tokens + estimate_message_tokens(trailing)

        self.assertEqual(
            second_request.context_view["estimated_tokens"],
            expected,
        )
        fallback_per_message_sum = (
            estimate_message_tokens(state.messages[0])
            + estimate_message_tokens(assistants[0])
            + estimate_message_tokens(trailing)
        )
        self.assertNotEqual(
            expected,
            fallback_per_message_sum,
            "test must distinguish usage-baseline accounting from per-message sum",
        )


if __name__ == "__main__":
    unittest.main()
