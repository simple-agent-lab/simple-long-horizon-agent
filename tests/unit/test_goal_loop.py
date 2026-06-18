"""Goal loop unit tests (deterministic; no network)."""

from __future__ import annotations

import unittest

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import AssistantMessage
from simple_agent_lab.protocols import GoalStatusEvent
from simple_agent_lab.workflow import (
    CompletionResult,
    GoalBudgets,
    run_goal_loop,
)

FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def _final_agent(name: str = "goal_agent") -> Agent:
    """An agent whose every turn emits a `final` message (one inner turn)."""

    def generate(messages):
        return AssistantMessage(content=(), sender=name, target="user", kind="final")

    return Agent(name=name, generate=generate)


def _fails_n_then_passes(n: int):
    calls = {"i": 0}

    def check(state):
        done = calls["i"] >= n
        calls["i"] += 1
        return CompletionResult(done=done)

    return check


class GoalLoopPhase1Test(unittest.TestCase):
    def test_check_passes_after_n_continuations_reports_complete(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "do the thing",
            check=_fails_n_then_passes(2),
            budgets=GoalBudgets(max_turns=10),
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.turns_used, 2)

    def test_check_never_passes_reports_budget_exhausted(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "do the thing",
            check=lambda s: CompletionResult(done=False),
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.turns_used, 3)

    def test_goal_status_recorded_as_events(self):
        agent = _final_agent()
        # capture the state by passing a check that records it
        seen = {}

        def check(state):
            seen["state"] = state
            return CompletionResult(done=True)

        run_goal_loop(agent, "obj", check=check)
        goal_events = [e for e in seen["state"].events if isinstance(e, GoalStatusEvent)]
        # one "active" (first run) + one terminal "complete"
        self.assertEqual([e.status for e in goal_events], ["active", "complete"])
        self.assertTrue(all(e.objective == "obj" for e in goal_events))
        # goal state is NOT parked in the mutable scratchpad
        self.assertNotIn("goal", seen["state"].data)

    def test_goal_timeline_survives_replay(self):
        # event-sourced: rebuilding the snapshot from the log preserves the
        # goal events (they ride in state.events, not the snapshot).
        agent = _final_agent()
        seen = {}

        def check(state):
            seen["state"] = state
            return CompletionResult(done=True)

        run_goal_loop(agent, "obj", check=check)
        state = seen["state"]
        before = [e.status for e in state.events if isinstance(e, GoalStatusEvent)]
        state.rebuild_snapshot()
        after = [e.status for e in state.events if isinstance(e, GoalStatusEvent)]
        self.assertEqual(before, after)

    def test_each_turn_produces_a_step(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "obj",
            check=_fails_n_then_passes(2),
            budgets=GoalBudgets(max_turns=10),
        )
        # first run + 2 continuations
        self.assertEqual(len(result.steps), 3)


if __name__ == "__main__":
    unittest.main()
