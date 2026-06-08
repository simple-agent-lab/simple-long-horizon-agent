"""Per-round model switching for LLM-backed agents.

`make_llm_agent` accepts either a single `Provider` (one model for the whole
run) or a map of named models paired with a `choose_model(ctx) -> name`
function that picks which model serves each round. These tests pin the
behavior with the deterministic fake adapter, which echoes `provider.model`
onto the response so the served model is observable on each assistant message.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from simple_agent_lab import AssistantMessage, RoundContext
from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import _provider_for_round, make_llm_agent
from simple_agent_lab.messages import (
    TextBlock,
    ToolResultBlock,
    assistant_message,
    tool_results_message,
    user_message,
)

ROOT = Path(__file__).resolve().parents[2]

FAST = Provider(id="fast", api="fake", model="fast-model")
STRONG = Provider(id="strong", api="fake", model="strong-model")
MODELS = {"fast": FAST, "strong": STRONG}


# A chooser that escalates by round: round 0 → fast, every later round → strong.
def escalate_by_round(ctx: RoundContext) -> str:
    return "fast" if ctx.round == 0 else "strong"


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


def _run_agent(agent):
    """Drive `agent` through the two-round bash task and return the final state."""
    state, events = agent.run(TWO_ROUND_TASK, max_turns=3)
    for _ in events:
        pass
    return state


def _run_bash(provider, choose_model=None):
    """Build a bash agent for `provider` and run it once."""
    return _run_agent(make_bash_agent(provider, cwd=ROOT, choose_model=choose_model))


class RoundContextTest(unittest.TestCase):
    def test_last_failed_reflects_the_most_recent_tool_result(self) -> None:
        ok = tool_results_message(
            [
                ToolResultBlock(
                    "c1", "bash", content=(TextBlock("done"),), is_error=False
                )
            ],
            target="a",
        )
        bad = tool_results_message(
            [
                ToolResultBlock(
                    "c2", "bash", content=(TextBlock("boom"),), is_error=True
                )
            ],
            target="a",
        )
        self.assertFalse(RoundContext(round=1, messages=(ok,)).last_failed)
        self.assertTrue(RoundContext(round=1, messages=(ok, bad)).last_failed)
        # No tool results at all → not a failure.
        self.assertFalse(RoundContext(round=0, messages=()).last_failed)


class ProviderResolutionTest(unittest.TestCase):
    def test_bare_provider_is_used_as_is(self) -> None:
        # A bare Provider is returned regardless of how far the run has gone.
        self.assertIs(_provider_for_round(FAST, None, [], "a"), FAST)
        self.assertIs(_provider_for_round(FAST, None, _prior("a", 3), "a"), FAST)

    def test_chooser_gets_round_index_from_this_agents_prior_messages(self) -> None:
        seen: list[int] = []

        def choose(ctx: RoundContext) -> str:
            seen.append(ctx.round)
            return escalate_by_round(ctx)

        # Round index = count of this agent's own prior messages; "b"'s are ignored.
        visible = [user_message("task", sender="user", target="a"), *_prior("b", 5)]
        self.assertIs(_provider_for_round(MODELS, choose, visible, "a"), FAST)
        self.assertIs(_provider_for_round(MODELS, choose, _prior("a", 1), "a"), STRONG)
        self.assertEqual(seen, [0, 1])

    def test_chooser_can_route_on_conversation_state(self) -> None:
        def on_failure_escalate(ctx: RoundContext) -> str:
            return "strong" if ctx.last_failed else "fast"

        bad = tool_results_message(
            [
                ToolResultBlock(
                    "c1", "bash", content=(TextBlock("boom"),), is_error=True
                )
            ],
            target="a",
        )
        self.assertIs(_provider_for_round(MODELS, on_failure_escalate, [], "a"), FAST)
        self.assertIs(
            _provider_for_round(MODELS, on_failure_escalate, [bad], "a"), STRONG
        )

    def test_unknown_key_raises_with_a_helpful_message(self) -> None:
        with self.assertRaises(KeyError) as caught:
            _provider_for_round(MODELS, lambda ctx: "huge", [], "a")
        self.assertIn("huge", str(caught.exception))


class MakeLlmAgentValidationTest(unittest.TestCase):
    def test_single_provider_rejects_a_chooser(self) -> None:
        with self.assertRaises(ValueError):
            make_llm_agent(name="a", provider=FAST, choose_model=escalate_by_round)

    def test_model_map_requires_a_chooser(self) -> None:
        with self.assertRaises(ValueError):
            make_llm_agent(name="a", provider=MODELS)

    def test_empty_model_map_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_llm_agent(name="a", provider={}, choose_model=escalate_by_round)


class PerRoundModelRunTest(unittest.TestCase):
    def test_single_provider_pins_one_model_across_rounds(self) -> None:
        state = _run_bash(FAST)
        models = _assistant_models(state, "bash_agent")
        self.assertEqual(len(models), 2)  # step + final
        self.assertEqual(models, ["fast-model", "fast-model"])

    def test_chooser_switches_model_each_round(self) -> None:
        state = _run_bash(MODELS, escalate_by_round)
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
        agent = make_bash_agent(MODELS, cwd=ROOT, choose_model=escalate_by_round)
        first = _run_agent(agent)
        second = _run_agent(agent)
        self.assertEqual(
            _assistant_models(first, "bash_agent"), ["fast-model", "strong-model"]
        )
        self.assertEqual(
            _assistant_models(second, "bash_agent"), ["fast-model", "strong-model"]
        )


if __name__ == "__main__":
    unittest.main()
