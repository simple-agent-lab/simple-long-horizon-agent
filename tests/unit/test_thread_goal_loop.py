"""Codex-style thread goal loop tests (deterministic; no network)."""

from __future__ import annotations

import unittest

from simple_agent_lab.core import Agent
from simple_agent_lab.messages import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    TokenUsage,
)
from simple_agent_lab.workflow import GoalBudgets
from simple_agent_lab.workflow.thread_goal_loop import (
    ThreadGoalStore,
    make_get_goal_tool,
    make_update_goal_tool,
    run_thread_goal_loop,
)


def _final_agent(name: str = "goal_agent") -> Agent:
    def generate(messages):
        return AssistantMessage(
            content=(TextBlock("still working"),),
            sender=name,
            target="user",
            kind="final",
        )

    return Agent(name=name, generate=generate)


class ThreadGoalStoreTest(unittest.TestCase):
    def test_replacing_unfinished_goal_requires_explicit_clear_or_terminal_status(self):
        store = ThreadGoalStore()
        first = store.create_goal("first objective")

        with self.assertRaisesRegex(ValueError, "unfinished goal"):
            store.create_goal("second objective")

        store.update_goal(first.goal_id, status="complete")
        second = store.create_goal("second objective")

        self.assertNotEqual(first.goal_id, second.goal_id)
        self.assertEqual(second.status, "active")

    def test_update_goal_accepts_only_model_terminal_statuses(self):
        store = ThreadGoalStore()
        goal = store.create_goal("ship the work")

        with self.assertRaisesRegex(ValueError, "complete or blocked"):
            store.update_goal(goal.goal_id, status="paused")

        self.assertEqual(store.get_goal(goal.goal_id).status, "active")


class ThreadGoalToolsTest(unittest.TestCase):
    def test_get_goal_tool_returns_current_goal_state(self):
        store = ThreadGoalStore()
        goal = store.create_goal("inspect state", token_budget=100)
        store.account_turn(goal.goal_id, tokens_used=30)
        tool = make_get_goal_tool(store, goal.goal_id)

        result = tool.execute("call_1", {}, lambda: False, None)

        self.assertFalse(result.is_error)
        self.assertIn("inspect state", result.content[0].text)
        self.assertEqual(result.details["goal"]["remaining_tokens"], 70)

    def test_update_goal_tool_mutates_goal_and_terminates_turn(self):
        store = ThreadGoalStore()
        goal = store.create_goal("finish statefully")
        tool = make_update_goal_tool(store, goal.goal_id)

        result = tool.execute(
            "call_1",
            {"status": "complete", "reason": "tests passed"},
            lambda: False,
            None,
        )

        self.assertTrue(result.terminate)
        self.assertFalse(result.is_error)
        updated = store.get_goal(goal.goal_id)
        self.assertEqual(updated.status, "complete")
        self.assertEqual(updated.reason, "tests passed")

    def test_update_goal_tool_rejects_lifecycle_statuses_owned_by_host(self):
        store = ThreadGoalStore()
        goal = store.create_goal("do not let model pause")
        tool = make_update_goal_tool(store, goal.goal_id)

        result = tool.execute("call_1", {"status": "paused"}, lambda: False, None)

        self.assertTrue(result.is_error)
        self.assertFalse(result.terminate)
        self.assertEqual(store.get_goal(goal.goal_id).status, "active")

    def test_goal_tools_can_be_bound_before_goal_exists(self):
        store = ThreadGoalStore()
        read_tool = make_get_goal_tool(store)
        update_tool = make_update_goal_tool(store)
        goal = store.create_goal("finish through prebound tools")

        read = read_tool.execute("call_read", {}, lambda: False, None)
        updated = update_tool.execute(
            "call_done",
            {"status": "complete", "reason": "verified"},
            lambda: False,
            None,
        )

        self.assertFalse(read.is_error)
        self.assertIn(goal.goal_id, read.content[0].text)
        self.assertFalse(updated.is_error)
        self.assertTrue(updated.terminate)
        self.assertEqual(store.get_goal(goal.goal_id).status, "complete")


class ThreadGoalLoopTest(unittest.TestCase):
    def test_loop_continues_until_model_updates_goal_complete(self):
        calls = {"n": 0}
        seen_tasks: list[str] = []

        def generate(messages):
            calls["n"] += 1
            seen_tasks.append(messages[-1].content[0].text)
            if calls["n"] < 3:
                return AssistantMessage(
                    content=(TextBlock(f"pass {calls['n']}"),),
                    sender="goal_agent",
                    target="user",
                    kind="final",
                )
            return AssistantMessage(
                content=(
                    TextBlock("verified done"),
                    ToolCallBlock(
                        "call_done",
                        "update_goal",
                        {"status": "complete", "reason": "verified"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent("goal_agent", generate)

        result = run_thread_goal_loop(
            agent,
            "keep the original objective",
            budgets=GoalBudgets(max_turns=5),
        )

        self.assertEqual(result.goal.status, "complete")
        self.assertEqual(result.goal.reason, "verified")
        self.assertEqual(result.turns_used, 3)
        self.assertEqual(calls["n"], 3)
        self.assertTrue(
            all("keep the original objective" in task for task in seen_tasks)
        )

    def test_loop_stops_when_model_updates_goal_blocked(self):
        def generate(messages):
            del messages
            return AssistantMessage(
                content=(
                    TextBlock("blocked"),
                    ToolCallBlock(
                        "call_blocked",
                        "update_goal",
                        {"status": "blocked", "reason": "missing credentials"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent("goal_agent", generate)

        result = run_thread_goal_loop(
            agent,
            "needs an external account",
            budgets=GoalBudgets(max_turns=5),
        )

        self.assertEqual(result.goal.status, "blocked")
        self.assertEqual(result.goal.reason, "missing credentials")
        self.assertEqual(result.stop_reason, "blocked")
        self.assertEqual(result.turns_used, 1)

    def test_loop_marks_budget_limited_when_turn_budget_is_exhausted(self):
        agent = _final_agent()

        result = run_thread_goal_loop(
            agent,
            "never completed",
            budgets=GoalBudgets(max_turns=2),
        )

        self.assertEqual(result.goal.status, "budget_limited")
        self.assertEqual(result.stop_reason, "budget_limited")
        self.assertEqual(result.turns_used, 2)

    def test_loop_automatically_adds_goal_tools_without_mutating_original_agent(self):
        seen_tool_names: list[tuple[str, ...]] = []

        def generate(messages):
            del messages
            seen_tool_names.append(tuple(tool.name for tool in agent.tools))
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent("goal_agent", generate)

        result = run_thread_goal_loop(agent, "finish via bound tools")

        self.assertEqual(result.goal.status, "complete")
        self.assertEqual(agent.tools, ())
        self.assertEqual(seen_tool_names, [()])

    def test_token_budget_updates_goal_usage_and_budget_status(self):
        def generate(messages):
            del messages
            return AssistantMessage(
                content=(TextBlock("used tokens"),),
                sender="goal_agent",
                target="user",
                kind="final",
                usage=TokenUsage(output_tokens=60),
            )

        agent = Agent("goal_agent", generate)

        result = run_thread_goal_loop(
            agent,
            "watch budget",
            budgets=GoalBudgets(max_turns=5, token_budget=100),
        )

        self.assertEqual(result.goal.status, "budget_limited")
        self.assertGreaterEqual(result.goal.tokens_used, 100)


if __name__ == "__main__":
    unittest.main()
