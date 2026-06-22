from __future__ import annotations

import unittest

from simple_agent_lab import Agent, Message, assistant_message, text_of
from simple_agent_lab.state import State
from simple_agent_lab.workflow import CompletionResult, pick_index, run_rtv


def _task_text(visible: list[Message]) -> str:
    task_msg = next(message for message in visible if message.kind == "task")
    return text_of(task_msg.content)


def make_fake_agent(name: str, reply):
    """One-turn, tool-free agent whose final answer is `reply(task_text)`."""

    def generate(visible: list[Message]) -> Message:
        return assistant_message(
            reply(_task_text(visible)), sender=name, target="user", kind="final"
        )

    return Agent(name, generate, role=f"{name} role")


def _const(value: str):
    return lambda task: value


def _check_for(target: str):
    """A `CompletionCheck` that is `done` once `target` appears in the state."""

    def check(state: State) -> CompletionResult:
        done = any(target in text_of(message.content) for message in state.messages)
        return CompletionResult(done=done)

    return check


class PickIndexTest(unittest.TestCase):
    def test_plain_number_is_one_based(self) -> None:
        self.assertEqual(pick_index("2", 3), 1)
        self.assertEqual(pick_index("Candidate 3 is best", 3), 2)

    def test_json_object_form(self) -> None:
        self.assertEqual(pick_index('{"best": 2}', 3), 1)
        self.assertEqual(pick_index('{"choice": 1}', 3), 0)

    def test_out_of_range_falls_back_to_default(self) -> None:
        self.assertEqual(pick_index("9", 3), 0)
        self.assertEqual(pick_index("9", 3, default=2), 2)

    def test_unparseable_falls_back(self) -> None:
        self.assertEqual(pick_index("the second one", 3), 0)

    def test_requires_positive_n(self) -> None:
        with self.assertRaises(ValueError):
            pick_index("1", 0)


class RtvTest(unittest.TestCase):
    def _workers(self) -> list[Agent]:
        return [
            make_fake_agent("w0", _const("AAA")),
            make_fake_agent("w1", _const("BBB")),
            make_fake_agent("w2", _const("CCC")),
            make_fake_agent("w3", _const("DDD")),
        ]

    def test_single_group_selects_named_candidate(self) -> None:
        # One group (group_size >= rollouts): selector picks candidate 2 -> w1.
        selector = make_fake_agent("sel", _const("2"))
        result = run_rtv(self._workers(), selector, "q", group_size=4)

        self.assertEqual(result.output, "BBB")
        # 4 rollouts (worker role) + exactly 1 selector comparison.
        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 4)
        self.assertEqual(roles.count("selector"), 1)

    def test_multi_round_tournament_reduces_to_one(self) -> None:
        # group_size=2 over 4 rollouts: round1 = 2 comparisons, round2 = 1.
        selector = make_fake_agent("sel", _const("1"))  # always first of group
        result = run_rtv(self._workers(), selector, "q", group_size=2)

        self.assertEqual(result.output, "AAA")  # index 0 survives
        self.assertEqual(sum(1 for s in result.steps if s.role == "selector"), 3)

    def test_output_is_full_rollout_not_summary(self) -> None:
        summarizer = make_fake_agent("sum", lambda answer_task: f"SUM:{answer_task}")
        selector = make_fake_agent("sel", _const("1"))
        result = run_rtv(
            self._workers(), selector, "q", group_size=4, summarizer=summarizer
        )

        # Selection runs over summaries, but the result is the winning rollout's
        # full answer, never the summary text.
        self.assertEqual(result.output, "AAA")
        self.assertEqual(sum(1 for s in result.steps if s.role == "summarizer"), 4)

    def test_single_agent_is_replicated_into_rollouts(self) -> None:
        worker = make_fake_agent("w", _const("ONE"))
        selector = make_fake_agent("sel", _const("1"))
        result = run_rtv(worker, selector, "q", rollouts=3, group_size=3)

        self.assertEqual(sum(1 for s in result.steps if s.role == "worker"), 3)
        self.assertEqual(result.output, "ONE")

    def test_one_rollout_skips_tournament(self) -> None:
        worker = make_fake_agent("w", _const("ONLY"))
        selector = make_fake_agent("sel", _const("boom"))  # must never run
        result = run_rtv(worker, selector, "q", rollouts=1)

        self.assertEqual(result.output, "ONLY")
        self.assertEqual(sum(1 for s in result.steps if s.role == "selector"), 0)

    def test_check_short_circuits_before_tournament(self) -> None:
        # A verified rollout (output "CCC") wins outright: no summaries, no
        # selector comparisons run, even with group_size that would tournament.
        selector = make_fake_agent("sel", _const("boom"))  # must never run
        result = run_rtv(
            self._workers(), selector, "q", group_size=2, check=_check_for("CCC")
        )

        self.assertEqual(result.output, "CCC")
        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 4)  # rollouts still fan out
        self.assertEqual(roles.count("selector"), 0)
        self.assertEqual(roles.count("summarizer"), 0)

    def test_check_that_never_passes_runs_full_tournament(self) -> None:
        # check never done -> falls back to the normal selector tournament.
        selector = make_fake_agent("sel", _const("1"))
        result = run_rtv(
            self._workers(), selector, "q", group_size=2, check=_check_for("ZZZ")
        )

        self.assertEqual(result.output, "AAA")
        self.assertEqual(sum(1 for s in result.steps if s.role == "selector"), 3)

    def test_empty_worker_sequence_raises(self) -> None:
        selector = make_fake_agent("sel", _const("1"))
        with self.assertRaises(ValueError):
            run_rtv([], selector, "q")


if __name__ == "__main__":
    unittest.main()
