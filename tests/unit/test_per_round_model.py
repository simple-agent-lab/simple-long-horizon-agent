"""Per-round (per-turn) model switching for LLM-backed agents.

`make_llm_agent` accepts either a single `Provider` (one model for the
whole run) or a `ProviderSelector` — a `(turn) -> Provider` callable
resolved before each model step. These tests pin the behavior with the
deterministic fake adapter, which echoes `provider.model` onto the
response so the served model is observable on each assistant message.
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

# Pin a known bash command so the fake takes exactly two turns:
# turn 0 emits the tool call (a `step`), turn 1 answers (the `final`).
TWO_TURN_TASK = "Use bash to run command: `echo per-round`"


def _assistant_models(state, name: str) -> list[str]:
    return [
        message.model
        for message in state.messages
        if isinstance(message, AssistantMessage) and message.sender == name
    ]


class ProviderResolutionTest(unittest.TestCase):
    def test_bare_provider_ignores_turn_and_is_used_as_is(self) -> None:
        two_prior = [
            assistant_message("a1", sender="a", target="user", kind="step"),
            assistant_message("a2", sender="a", target="user", kind="step"),
        ]
        # A bare Provider is returned regardless of how far the run has gone.
        self.assertIs(_provider_for_turn(FAST, [], "a"), FAST)
        self.assertIs(_provider_for_turn(FAST, two_prior, "a"), FAST)

    def test_selector_gets_turn_index_from_this_agents_prior_messages(self) -> None:
        seen: list[int] = []

        def selector(turn: int) -> Provider:
            seen.append(turn)
            return FAST if turn == 0 else STRONG

        visible = [
            user_message("task", sender="user", target="a"),
            assistant_message("a1", sender="a", target="user", kind="step"),
            assistant_message("b1", sender="b", target="user", kind="step"),
        ]
        # One prior assistant message from "a" → turn 1; "b"'s message is ignored.
        self.assertIs(_provider_for_turn(selector, visible, "a"), STRONG)
        # No prior messages from "a" yet → turn 0.
        self.assertIs(_provider_for_turn(selector, [], "a"), FAST)
        self.assertEqual(seen, [1, 0])


class PerRoundModelRunTest(unittest.TestCase):
    def test_single_provider_pins_one_model_across_turns(self) -> None:
        agent = make_bash_agent(FAST, cwd=ROOT)
        state, events = agent.run(TWO_TURN_TASK, max_turns=3)
        for _ in events:
            pass

        models = _assistant_models(state, "bash_agent")
        self.assertEqual(len(models), 2)  # step + final
        self.assertEqual(models, ["fast-model", "fast-model"])

    def test_selector_switches_model_each_turn(self) -> None:
        seen_turns: list[int] = []

        def selector(turn: int) -> Provider:
            seen_turns.append(turn)
            return FAST if turn == 0 else STRONG

        agent = make_bash_agent(selector, cwd=ROOT)
        state, events = agent.run(TWO_TURN_TASK, max_turns=3)
        for _ in events:
            pass

        # The selector saw consecutive turn indices, one per model step.
        self.assertEqual(seen_turns, [0, 1])
        # Turn 0 ran on the fast model; turn 1 escalated to the strong model —
        # visible both on the assistant messages and on the response events.
        self.assertEqual(
            _assistant_models(state, "bash_agent"),
            ["fast-model", "strong-model"],
        )
        response_models = [
            event.model for event in state.events if event.kind == "model_response"
        ]
        self.assertEqual(response_models, ["fast-model", "strong-model"])

    def test_selector_resets_per_run_no_cross_run_leak(self) -> None:
        # The turn index is derived from the visible context, so a reused agent
        # starts at turn 0 again on its next run rather than carrying a counter.
        agent = make_bash_agent(lambda turn: FAST if turn == 0 else STRONG, cwd=ROOT)

        first, first_events = agent.run(TWO_TURN_TASK, max_turns=3)
        for _ in first_events:
            pass
        second, second_events = agent.run(TWO_TURN_TASK, max_turns=3)
        for _ in second_events:
            pass

        self.assertEqual(
            _assistant_models(first, "bash_agent"), ["fast-model", "strong-model"]
        )
        self.assertEqual(
            _assistant_models(second, "bash_agent"), ["fast-model", "strong-model"]
        )

    def test_cycling_list_expressed_as_a_selector(self) -> None:
        models = [FAST, STRONG]
        agent = make_llm_agent(
            name="cycler",
            provider=lambda turn: models[turn % len(models)],
        )
        # No tools, so the fake ends the turn immediately: one model step.
        state, events = agent.run("hello", max_turns=3)
        for _ in events:
            pass
        self.assertEqual(_assistant_models(state, "cycler"), ["fast-model"])


if __name__ == "__main__":
    unittest.main()
