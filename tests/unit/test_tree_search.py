from __future__ import annotations

import re
import unittest

from simple_agent_lab import Agent, Message, assistant_message, text_of
from simple_agent_lab.workflow import (
    emitted_final,
    fork_state,
    resume_agent,
    run_agent,
    run_mcts,
)


def _task_text(visible: list[Message]) -> str:
    task_msg = next(message for message in visible if message.kind == "task")
    return text_of(task_msg.content)


def make_fake_agent(name: str, reply):
    def generate(visible: list[Message]) -> Message:
        return assistant_message(
            reply(_task_text(visible)), sender=name, target="user", kind="final"
        )

    return Agent(name, generate, role=f"{name} role")


def make_counting_agent(name: str, prefix: str):
    state = {"n": 0}

    def reply(task: str) -> str:
        del task
        state["n"] += 1
        return f"{prefix}{state['n']}"

    return make_fake_agent(name, reply)


def _const(value: str):
    return lambda task: value


def _value_by_trailing_digit(task: str) -> str:
    """Score 'ansN' as 0.N — so higher-numbered rollouts win."""
    match = re.search(r"ans(\d+)", task)
    return f"0.{match.group(1)}" if match else "0.0"


class BaseHelpersTest(unittest.TestCase):
    def test_fork_state_is_independent(self) -> None:
        worker = make_fake_agent("w", _const("hi"))
        step = run_agent(worker, "task")
        forked = fork_state(step.state)

        self.assertIsNot(forked, step.state)
        self.assertIsNot(forked.events, step.state.events)
        self.assertEqual(len(forked.messages), len(step.state.messages))

        before = len(step.state.events)
        resume_agent(worker, forked, "again")
        # Growing the fork must not mutate the parent state.
        self.assertEqual(len(step.state.events), before)
        self.assertGreater(len(forked.events), before)

    def test_emitted_final_detects_terminal(self) -> None:
        worker = make_fake_agent("w", _const("done"))
        step = run_agent(worker, "t")
        self.assertTrue(emitted_final(step.state, "w"))

        def chatty(visible: list[Message]) -> Message:
            del visible
            return assistant_message("x", sender="c", target="user", kind="step")

        chatty_step = run_agent(Agent("c", chatty), "t", max_turns=1)
        self.assertFalse(emitted_final(chatty_step.state, "c"))


class MctsTest(unittest.TestCase):
    def test_expands_branch_children_and_scores_each(self) -> None:
        worker = make_fake_agent("w", _const("ans"))
        value = make_fake_agent("v", _const("0.5"))
        result = run_mcts(worker, value, "q", budget=4, branch=4, segment_turns=1)

        roles = [s.role for s in result.steps]
        self.assertEqual(roles.count("rollout"), 4)
        self.assertEqual(roles.count("value"), 4)
        self.assertEqual(result.output, "ans")

    def test_picks_highest_value_terminal(self) -> None:
        worker = make_counting_agent("w", "ans")  # ans1, ans2, ans3, ans4
        value = make_fake_agent("v", _value_by_trailing_digit)
        result = run_mcts(worker, value, "q", budget=4, branch=4, segment_turns=1)

        self.assertEqual(result.output, "ans4")  # 0.4 is the best score

    def test_budget_bounds_expansions(self) -> None:
        worker = make_counting_agent("w", "ans")
        value = make_fake_agent("v", _value_by_trailing_digit)
        result = run_mcts(worker, value, "q", budget=2, branch=4, segment_turns=1)

        self.assertEqual(sum(1 for s in result.steps if s.role == "rollout"), 2)
        self.assertEqual(result.output, "ans2")

    def test_invalid_budget_or_branch_raise(self) -> None:
        worker = make_fake_agent("w", _const("a"))
        value = make_fake_agent("v", _const("0.1"))
        with self.assertRaises(ValueError):
            run_mcts(worker, value, "q", budget=0)
        with self.assertRaises(ValueError):
            run_mcts(worker, value, "q", branch=0)


if __name__ == "__main__":
    unittest.main()
