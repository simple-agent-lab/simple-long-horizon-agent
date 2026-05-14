from __future__ import annotations

import unittest

from simple_agent_lab import (
    Agent,
    AssistantMessage,
    Message,
    ModelRequestEvent,
    State,
    TurnEndEvent,
    estimate_message_tokens,
    make_llm_step,
    run,
)
from simple_agent_lab.llm import Provider


class ContextUsageTest(unittest.TestCase):
    def test_next_turn_uses_latest_usage_plus_trailing_message_estimate(self) -> None:
        provider = Provider(id="fake-context-usage", api="fake", model="fake-model")
        agent = Agent(
            "writer",
            make_llm_step(
                provider,
                system_prompt="Answer briefly.",
                target="user",
            ),
        )
        state = State(
            "Explain why context accounting should use provider usage when available."
        )
        state.send("task", "user", "writer", state.task)

        trailing: Message | None = None
        schedule = iter(["writer", "writer"])
        for event in run([agent], state, lambda _: next(schedule, None)):
            if isinstance(event, TurnEndEvent) and trailing is None:
                trailing = state.send(
                    "note",
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
        first_usage = assistants[0].usage
        self.assertIsNotNone(first_usage)
        assert first_usage is not None

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
