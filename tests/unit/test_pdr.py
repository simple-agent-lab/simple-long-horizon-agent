from __future__ import annotations

import unittest

from simple_long_horizon_agent import Agent, Message, assistant_message, text_of
from simple_long_horizon_agent.state import State
from simple_long_horizon_agent.workflow import CompletionResult, run_pdr


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
    """Sequential-only fake whose Nth reply is f'{prefix}{N}' (not thread-safe)."""
    state = {"n": 0}

    def reply(task: str) -> str:
        del task
        state["n"] += 1
        return f"{prefix}{state['n']}"

    return make_fake_agent(name, reply)


def _check_for(target: str):
    """A `CompletionCheck` that is `done` once `target` appears in the state."""

    def check(state: State) -> CompletionResult:
        done = any(target in text_of(message.content) for message in state.messages)
        return CompletionResult(done=done)

    return check


class PdrTest(unittest.TestCase):
    def test_step_structure_and_roles(self) -> None:
        worker = make_fake_agent("attempt", lambda task: "a")
        distiller = make_fake_agent("distiller", lambda task: "brief")
        finalizer = make_fake_agent("finalizer", lambda task: "final answer")
        result = run_pdr(worker, distiller, "q", rounds=2, width=2, finalizer=finalizer)

        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 4)  # 2 rounds * width 2
        self.assertEqual(roles.count("distiller"), 2)
        self.assertEqual(roles.count("finalizer"), 1)
        self.assertEqual(result.output, "final answer")

    def test_brief_threads_into_next_round_and_finalizer(self) -> None:
        worker = make_fake_agent("attempt", lambda task: "a")
        distiller = make_counting_agent(
            "distiller", "BRIEF"
        )  # sequential: BRIEF1, BRIEF2
        finalizer = make_fake_agent("finalizer", lambda task: "done")
        result = run_pdr(worker, distiller, "q", rounds=2, width=2, finalizer=finalizer)

        attempts = [s for s in result.steps if s.role == "worker"]
        # Round-2 attempts (3rd and 4th) are conditioned on round-1's brief.
        self.assertIn("BRIEF1", attempts[2].task)
        self.assertIn("<prior_findings>", attempts[2].task)
        # Round-1 attempts saw no prior findings.
        self.assertNotIn("prior_findings", attempts[0].task)
        # The finalizer is conditioned on the latest (round-2) brief.
        finalizer_step = next(s for s in result.steps if s.role == "finalizer")
        self.assertIn("BRIEF2", finalizer_step.task)

    def test_finalizer_defaults_to_worker(self) -> None:
        worker = make_fake_agent("attempt", lambda task: "ans")
        distiller = make_fake_agent("distiller", lambda task: "brief")
        result = run_pdr(worker, distiller, "q", rounds=1, width=1)

        finalizer_step = next(s for s in result.steps if s.role == "finalizer")
        self.assertEqual(finalizer_step.name, "attempt")
        self.assertEqual(result.output, "ans")

    def test_check_short_circuits_after_first_round(self) -> None:
        # Round-1 attempts already pass the check -> stop before distill,
        # remaining rounds, and the finalizer.
        worker = make_fake_agent("attempt", lambda task: "SOLVED")
        distiller = make_fake_agent("distiller", lambda task: "brief")  # never runs
        finalizer = make_fake_agent("finalizer", lambda task: "final")  # never runs
        result = run_pdr(
            worker,
            distiller,
            "q",
            rounds=3,
            width=2,
            finalizer=finalizer,
            check=_check_for("SOLVED"),
        )

        self.assertEqual(result.output, "SOLVED")
        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 2)  # only round 1's attempts
        self.assertEqual(roles.count("distiller"), 0)
        self.assertEqual(roles.count("finalizer"), 0)

    def test_check_that_never_passes_runs_full_budget(self) -> None:
        worker = make_fake_agent("attempt", lambda task: "a")
        distiller = make_fake_agent("distiller", lambda task: "brief")
        finalizer = make_fake_agent("finalizer", lambda task: "final")
        result = run_pdr(
            worker,
            distiller,
            "q",
            rounds=2,
            width=2,
            finalizer=finalizer,
            check=_check_for("NOPE"),
        )

        self.assertEqual(result.output, "final")
        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 4)
        self.assertEqual(roles.count("distiller"), 2)
        self.assertEqual(roles.count("finalizer"), 1)

    def test_invalid_rounds_or_width_raise(self) -> None:
        worker = make_fake_agent("attempt", lambda task: "a")
        distiller = make_fake_agent("distiller", lambda task: "b")
        with self.assertRaises(ValueError):
            run_pdr(worker, distiller, "q", rounds=0)
        with self.assertRaises(ValueError):
            run_pdr(worker, distiller, "q", width=0)
