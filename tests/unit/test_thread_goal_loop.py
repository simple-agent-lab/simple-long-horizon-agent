"""Codex-style thread goal loop tests (deterministic; no network)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from simple_agent_lab import State, make_llm_agent, message_text
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.types import LLMRequest, LLMResponse
from simple_agent_lab.messages import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    TokenUsage,
)
from simple_agent_lab.protocols import GoalStatusEvent
from simple_agent_lab.workflow import GoalBudgets
from simple_agent_lab.workflow.thread_goal_loop import (
    THREAD_GOAL_STORE_DATA_KEY,
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

    def test_goal_lifecycle_is_recorded_in_state_events(self):
        calls = {"n": 0}

        def generate(messages):
            del messages
            calls["n"] += 1
            if calls["n"] == 1:
                return AssistantMessage(
                    content=(TextBlock("made progress"),),
                    sender="goal_agent",
                    target="user",
                    kind="final",
                )
            return AssistantMessage(
                content=(
                    TextBlock("done"),
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

        result = run_thread_goal_loop(
            Agent("goal_agent", generate),
            "record this goal in state",
            budgets=GoalBudgets(max_turns=5),
        )
        state = result.steps[-1].state
        goal_events = [e for e in state.events if isinstance(e, GoalStatusEvent)]

        self.assertEqual([e.status for e in goal_events], ["active", "complete"])
        self.assertTrue(
            all(e.objective == "record this goal in state" for e in goal_events)
        )
        self.assertTrue(all(e.goal_id == result.goal.goal_id for e in goal_events))
        self.assertEqual([e.turns_used for e in goal_events], [1, 2])
        self.assertNotIn("goal", state.data)

    def test_goal_store_is_attached_to_state_data(self):
        def generate(messages):
            del messages
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock(
                        "call_done",
                        "update_goal",
                        {"status": "complete", "reason": "stored in state"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        store = ThreadGoalStore()
        result = run_thread_goal_loop(
            Agent("goal_agent", generate),
            "make state carry the goal store",
            goal_store=store,
        )
        state = result.steps[-1].state

        self.assertIs(state.data[THREAD_GOAL_STORE_DATA_KEY], store)
        self.assertEqual(
            store.get_goal(result.goal.goal_id).reason,
            "stored in state",
        )
        self.assertFalse(hasattr(state, "resources"))

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

    def test_loop_adds_goal_tools_to_llm_request_without_mutating_agent(self):
        seen_tool_names: list[tuple[str, ...]] = []

        def complete(request: LLMRequest) -> LLMResponse:
            seen_tool_names.append(tuple(tool.name for tool in request.tools))
            return LLMResponse(
                content=(
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                stop_reason="tool_use",
                model=request.provider.model,
            )

        agent = make_llm_agent(
            name="goal_agent",
            provider=Provider(id="fake", api="fake", model="fake-model"),
        )

        with patch(
            "simple_agent_lab.llm_agent.complete_with_tool_call_retry", complete
        ):
            result = run_thread_goal_loop(agent, "finish through the injected tool")

        self.assertEqual(result.goal.status, "complete")
        self.assertEqual(seen_tool_names, [("get_goal", "update_goal")])
        self.assertEqual(agent.tools, ())

    def test_loop_rebinds_same_named_goal_tools_to_the_current_goal(self):
        stale_store = ThreadGoalStore()
        stale_goal = stale_store.create_goal("an older run")
        current_store = ThreadGoalStore()

        def generate(messages):
            del messages
            return AssistantMessage(
                content=(
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent(
            "goal_agent",
            generate,
            tools=(make_update_goal_tool(stale_store, stale_goal.goal_id),),
        )

        result = run_thread_goal_loop(
            agent,
            "finish the current run",
            budgets=GoalBudgets(max_turns=1),
            goal_store=current_store,
        )

        self.assertEqual(result.goal.status, "complete")
        self.assertEqual(stale_store.get_goal(stale_goal.goal_id).status, "active")
        self.assertEqual(agent.tools[0].name, "update_goal")

    def test_injected_state_resumes_and_inherits_prior_context(self):
        seed = _final_agent("goal_agent")
        seed_state, seed_events = seed.run("prior context marker")
        for _ in seed_events:
            pass
        prior_messages = len(seed_state.messages)

        seen_tasks: list[str] = []

        def generate(messages):
            seen_tasks.append(messages[-1].content[0].text)
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        result = run_thread_goal_loop(
            Agent("goal_agent", generate),
            "solve the current sub-problem",
            budgets=GoalBudgets(max_turns=3),
            state=seed_state,
            steering_preface="LONG CHAIN PREFACE",
        )

        # Resumes on the SAME state object, so it inherits accumulated context.
        self.assertIs(result.steps[-1].state, seed_state)
        self.assertGreater(len(seed_state.messages), prior_messages)
        text = "\n".join(message_text(m) for m in seed_state.messages)
        self.assertIn("prior context marker", text)
        # The steering carried the trusted long-chain preface, then the loop's
        # own steering body.
        self.assertTrue(seen_tasks[0].startswith("LONG CHAIN PREFACE"))
        self.assertIn("Continue working toward the active goal", seen_tasks[0])

    def test_injected_state_token_budget_excludes_prior_output_usage(self):
        state = State("shared chain")
        state.record(
            AssistantMessage(
                content=(TextBlock("earlier instance"),),
                sender="goal_agent",
                target="user",
                kind="final",
                usage=TokenUsage(output_tokens=100),
            )
        )

        def generate(messages):
            del messages
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
                usage=TokenUsage(output_tokens=1),
            )

        result = run_thread_goal_loop(
            Agent("goal_agent", generate),
            "new instance",
            budgets=GoalBudgets(max_turns=2, token_budget=50),
            state=state,
        )

        self.assertEqual(result.goal.status, "complete")
        self.assertEqual(result.goal.tokens_used, 1)

    def test_default_state_and_preface_keep_original_behavior(self):
        seen_tasks: list[str] = []

        def generate(messages):
            seen_tasks.append(messages[-1].content[0].text)
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock("call_done", "update_goal", {"status": "complete"}),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        result = run_thread_goal_loop(
            Agent("goal_agent", generate),
            "single instance objective",
        )

        self.assertEqual(result.goal.status, "complete")
        # No preface and no injected state: the steering begins exactly as before.
        self.assertTrue(seen_tasks[0].startswith("Continue working toward"))

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
