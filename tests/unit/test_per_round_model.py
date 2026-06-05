"""Per-round model switching for LLM-backed agents.

`make_llm_agent` accepts either a single `Provider` (one model for the
whole run) or a list of `Provider`s — one model per round, with the last
one sticking once the list runs out. These tests pin the behavior with the
deterministic fake adapter, which echoes `provider.model` onto the response
so the served model is observable on each assistant message.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simple_agent_lab import AssistantMessage
from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import _provider_for_turn, make_llm_agent
from simple_agent_lab.messages import assistant_message, user_message

ROOT = Path(__file__).resolve().parents[2]

FAST = Provider(id="fast", api="fake", model="fast-model")
STRONG = Provider(id="strong", api="fake", model="strong-model")

# Pin a known bash command so the fake takes exactly two rounds:
# round 0 emits the tool call (a `step`), round 1 answers (the `final`).
TWO_ROUND_TASK = "Use bash to run command: `echo per-round`"


def _assistant_models(state, name: str) -> list[str]:
    return [
        message.model
        for message in state.messages
        if isinstance(message, AssistantMessage) and message.sender == name
    ]


def _prior(name: str, count: int) -> list[AssistantMessage]:
    """`count` assistant messages already authored by `name`."""
    return [
        assistant_message(f"{name}{i}", sender=name, target="user", kind="step")
        for i in range(count)
    ]


class ProviderResolutionTest(unittest.TestCase):
    def test_bare_provider_is_used_as_is(self) -> None:
        # A bare Provider is returned regardless of how far the run has gone.
        self.assertIs(_provider_for_turn(FAST, [], "a"), FAST)
        self.assertIs(_provider_for_turn(FAST, _prior("a", 3), "a"), FAST)

    def test_list_picks_the_model_for_this_rounds_index(self) -> None:
        models = [FAST, STRONG]
        # Round index = count of this agent's own prior messages; "b"'s are ignored.
        visible = [user_message("task", sender="user", target="a"), *_prior("b", 5)]
        self.assertIs(_provider_for_turn(models, visible, "a"), FAST)  # round 0
        self.assertIs(
            _provider_for_turn(models, _prior("a", 1), "a"), STRONG
        )  # round 1

    def test_last_model_sticks_past_the_end_of_the_list(self) -> None:
        models = [FAST, STRONG]
        # Rounds 2 and 3 are past the list length, so the last model applies.
        self.assertIs(_provider_for_turn(models, _prior("a", 2), "a"), STRONG)
        self.assertIs(_provider_for_turn(models, _prior("a", 3), "a"), STRONG)


class PerRoundModelRunTest(unittest.TestCase):
    def test_single_provider_pins_one_model_across_rounds(self) -> None:
        agent = make_bash_agent(FAST, cwd=ROOT)
        state, events = agent.run(TWO_ROUND_TASK, max_turns=3)
        for _ in events:
            pass

        models = _assistant_models(state, "bash_agent")
        self.assertEqual(len(models), 2)  # step + final
        self.assertEqual(models, ["fast-model", "fast-model"])

    def test_list_switches_model_each_round(self) -> None:
        agent = make_bash_agent([FAST, STRONG], cwd=ROOT)
        state, events = agent.run(TWO_ROUND_TASK, max_turns=3)
        for _ in events:
            pass

        # Round 0 ran on the fast model; round 1 escalated to the strong model —
        # visible both on the assistant messages and on the response events.
        self.assertEqual(
            _assistant_models(state, "bash_agent"),
            ["fast-model", "strong-model"],
        )
        response_models = [
            event.model for event in state.events if event.kind == "model_response"
        ]
        self.assertEqual(response_models, ["fast-model", "strong-model"])

    def test_round_index_resets_per_run_no_cross_run_leak(self) -> None:
        # The round index is derived from the visible context, so a reused agent
        # starts at round 0 again on its next run rather than carrying a counter.
        agent = make_bash_agent([FAST, STRONG], cwd=ROOT)

        first, first_events = agent.run(TWO_ROUND_TASK, max_turns=3)
        for _ in first_events:
            pass
        second, second_events = agent.run(TWO_ROUND_TASK, max_turns=3)
        for _ in second_events:
            pass

        self.assertEqual(
            _assistant_models(first, "bash_agent"), ["fast-model", "strong-model"]
        )
        self.assertEqual(
            _assistant_models(second, "bash_agent"), ["fast-model", "strong-model"]
        )

    def test_empty_provider_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_llm_agent(name="agent", provider=[])


if __name__ == "__main__":
    unittest.main()
