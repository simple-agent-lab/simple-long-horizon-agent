"""Goal loop unit tests (deterministic; no network)."""

from __future__ import annotations

import unittest

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import AssistantMessage, TokenUsage
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


def _final_agent_with_usage(output_tokens: int, name: str = "goal_agent") -> Agent:
    """An agent whose every turn emits a `final` message with token usage."""

    def generate(messages):
        return AssistantMessage(
            content=(),
            sender=name,
            target="user",
            kind="final",
            usage=TokenUsage(output_tokens=output_tokens),
        )

    return Agent(name=name, generate=generate)


class GoalLoopPhase2Test(unittest.TestCase):
    def test_token_budget_hit_reports_budget_exhausted(self):
        # Each turn emits 50 output tokens; budget=120 → exhausted after 3 turns
        # (first run=50, resume1=100, resume2=150 >= 120).
        agent = _final_agent_with_usage(output_tokens=50)
        result = run_goal_loop(
            agent,
            "obj",
            check=lambda s: CompletionResult(done=False),
            budgets=GoalBudgets(token_budget=120),
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertGreaterEqual(result.tokens_used, 120)

    def test_same_blocker_three_turns_reports_blocked(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "obj",
            check=lambda s: CompletionResult(done=False, blocked=True, reason="no network"),
            budgets=GoalBudgets(max_turns=10),
        )
        self.assertEqual(result.status, "blocked")

    def test_changing_blocker_does_not_trip_streak(self):
        agent = _final_agent()
        reasons = iter(["a", "b", "c", "d"])
        result = run_goal_loop(
            agent,
            "obj",
            check=lambda s: CompletionResult(
                done=False, blocked=True, reason=next(reasons, "z")
            ),
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "budget_exhausted")  # never 3 in a row

    def test_caller_abort_reports_aborted(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "obj",
            check=lambda s: CompletionResult(done=False),
            abort=lambda: True,
        )
        self.assertEqual(result.status, "aborted")

    def test_wall_clock_deadline_reports_aborted(self):
        agent = _final_agent()
        result = run_goal_loop(
            agent,
            "obj",
            check=lambda s: CompletionResult(done=False),
            budgets=GoalBudgets(wall_clock_seconds=0.0),
        )
        self.assertEqual(result.status, "aborted")

    def test_tokens_used_in_goal_result(self):
        # Verify tokens_used is tracked and reported in GoalResult.
        agent = _final_agent_with_usage(output_tokens=30)
        result = run_goal_loop(
            agent,
            "obj",
            check=_fails_n_then_passes(1),
            budgets=GoalBudgets(max_turns=5),
        )
        self.assertEqual(result.status, "complete")
        # first run + 1 continuation = 2 turns * 30 tokens each = 60
        self.assertEqual(result.tokens_used, 60)

    def test_tokens_used_recorded_in_goal_status_events(self):
        # Verify GoalStatusEvent carries the tokens_used on each turn.
        agent = _final_agent_with_usage(output_tokens=25)
        seen = {}

        def check(state):
            seen["state"] = state
            return CompletionResult(done=True)

        run_goal_loop(agent, "obj", check=check)
        state = seen["state"]
        goal_events = [e for e in state.events if isinstance(e, GoalStatusEvent)]
        # All events should carry non-zero tokens_used (25 from first run).
        self.assertTrue(all(e.tokens_used >= 25 for e in goal_events))

    def test_non_blocked_verdict_resets_streak(self):
        # Alternate blocked/not-blocked — streak should never reach 3.
        agent = _final_agent()
        call_count = {"i": 0}

        def alternating_check(state):
            i = call_count["i"]
            call_count["i"] += 1
            if i % 2 == 0:
                return CompletionResult(done=False, blocked=True, reason="same blocker")
            return CompletionResult(done=False, blocked=False)

        result = run_goal_loop(
            agent,
            "obj",
            check=alternating_check,
            budgets=GoalBudgets(max_turns=4),
        )
        self.assertEqual(result.status, "budget_exhausted")  # streak never reaches 3


if __name__ == "__main__":
    unittest.main()
