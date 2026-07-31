from __future__ import annotations

import unittest

from simple_long_horizon_agent import Agent, Message, assistant_message, text_of
from simple_long_horizon_agent.workflow import (
    Route,
    run_chain,
    run_parallel,
    run_planner_executor,
    run_reflection,
    run_routing,
    select_route,
)
from simple_long_horizon_agent.workflow.base import final_output, run_agent


def _task_text(visible: list[Message]) -> str:
    """Full text of the seeded task message a sub-run sees."""
    task_msg = next(message for message in visible if message.kind == "task")
    return text_of(task_msg.content)


def make_fake_agent(name: str, reply):
    """Agent whose final answer is `reply(task_text)`; no tools, one turn."""

    def generate(visible: list[Message]) -> Message:
        return assistant_message(
            reply(_task_text(visible)),
            sender=name,
            target="user",
            kind="final",
        )

    return Agent(name, generate, role=f"{name} role")


class BaseTest(unittest.TestCase):
    def test_run_agent_extracts_final_output(self) -> None:
        agent = make_fake_agent("a", lambda task: f"answer:{task}")
        step = run_agent(agent, "hello")
        self.assertEqual(step.output, "answer:hello")
        self.assertEqual(step.name, "a")
        # The full run state is retained for tracing.
        self.assertTrue(step.state.events)

    def test_final_output_falls_back_to_last_assistant_on_truncation(self) -> None:
        def chatty(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "still going", sender="chatty", target="user", kind="step"
            )

        agent = Agent("chatty", chatty)
        step = run_agent(agent, "go", max_turns=2)
        # No `final` ever emitted; the last assistant utterance is used instead.
        self.assertEqual(step.output, "still going")
        self.assertEqual(final_output(step.state, "chatty"), "still going")


class SequentialTest(unittest.TestCase):
    def test_chain_threads_output_through_stages(self) -> None:
        # Each stage tags the text it received so we can see the carry-forward.
        first = make_fake_agent("first", lambda task: f"[first]{task}")
        second = make_fake_agent("second", lambda task: f"[second]{task}")
        result = run_chain([first, second], "seed")

        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].output, "[first]seed")
        # Second stage saw the original task + the first stage's output.
        self.assertIn("[first]seed", result.steps[1].task)
        self.assertIn("seed", result.steps[1].task)
        self.assertEqual(result.output, result.steps[1].output)

    def test_chain_requires_at_least_one_agent(self) -> None:
        with self.assertRaises(ValueError):
            run_chain([], "x")


class PlannerExecutorTest(unittest.TestCase):
    def test_plan_feeds_executor(self) -> None:
        planner = make_fake_agent("planner", lambda task: f"PLAN for {task}")
        executor = make_fake_agent("executor", lambda task: f"DID: {task}")
        result = run_planner_executor(planner, executor, "build it")

        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].role, "planner")
        self.assertEqual(result.steps[1].role, "executor")
        # The executor's task embeds both the original task and the plan.
        self.assertIn("PLAN for", result.steps[1].task)
        self.assertIn("build it", result.steps[1].task)
        self.assertEqual(result.output, result.steps[1].output)


class ReflectionTest(unittest.TestCase):
    def test_loops_until_critic_approves(self) -> None:
        def generator_reply(task: str) -> str:
            return "revised" if "Revise your draft" in task else "draft"

        approvals = {"n": 0}

        def critic_reply(task: str) -> str:
            del task
            approvals["n"] += 1
            # Approve only on the second critique.
            return "APPROVED" if approvals["n"] >= 2 else "needs work"

        generator = make_fake_agent("gen", generator_reply)
        critic = make_fake_agent("crit", critic_reply)
        result = run_reflection(generator, critic, "task", max_rounds=3)

        # draft, critique#1 (reject), revise, critique#2 (approve) = 4 steps.
        roles = [step.role for step in result.steps]
        self.assertEqual(roles, ["generator", "critic", "generator", "critic"])
        self.assertEqual(result.output, "revised")

    def test_stops_at_round_budget_without_approval(self) -> None:
        generator = make_fake_agent("gen", lambda task: "draft")
        critic = make_fake_agent("crit", lambda task: "never happy")
        result = run_reflection(generator, critic, "task", max_rounds=2)

        # draft + 2 * (critique + revise) = 5 steps, output is last revision.
        self.assertEqual(len(result.steps), 5)
        self.assertEqual(result.output, "draft")


class RoutingTest(unittest.TestCase):
    def test_routes_to_named_specialist(self) -> None:
        router = make_fake_agent("router", lambda task: "python")
        py = make_fake_agent("python", lambda task: f"py:{task}")
        sql = make_fake_agent("sql", lambda task: f"sql:{task}")
        routes = [Route("python", py), Route("sql", sql)]

        result = run_routing(router, routes, "loop over a list")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[1].name, "python")
        self.assertEqual(result.output, "py:loop over a list")

    def test_select_route_prefers_longest_match(self) -> None:
        a = make_fake_agent("sql", lambda t: t)
        b = make_fake_agent("sql_writer", lambda t: t)
        routes = [Route("sql", a), Route("sql_writer", b)]
        chosen = select_route("use the sql_writer please", routes)
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.name, "sql_writer")

    def test_unresolved_route_returns_router_step_only(self) -> None:
        router = make_fake_agent("router", lambda task: "nonsense")
        only = make_fake_agent("only", lambda task: "ran")
        result = run_routing(router, [Route("only", only)], "x")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].name, "router")

    def test_default_route_used_when_unresolved(self) -> None:
        router = make_fake_agent("router", lambda task: "nonsense")
        only = make_fake_agent("only", lambda task: "ran")
        result = run_routing(router, [Route("only", only)], "x", default="only")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.output, "ran")


class ParallelTest(unittest.TestCase):
    def test_workers_run_and_order_is_preserved(self) -> None:
        workers = [
            make_fake_agent("w0", lambda task: "alpha"),
            make_fake_agent("w1", lambda task: "beta"),
            make_fake_agent("w2", lambda task: "gamma"),
        ]
        result = run_parallel(workers, "same task")
        self.assertEqual([step.name for step in result.steps], ["w0", "w1", "w2"])
        self.assertIn("alpha", result.output)
        self.assertIn("gamma", result.output)

    def test_aggregator_synthesizes(self) -> None:
        workers = [
            make_fake_agent("w0", lambda task: "alpha"),
            make_fake_agent("w1", lambda task: "beta"),
        ]
        aggregator = make_fake_agent("agg", lambda task: "synthesized")
        result = run_parallel(workers, "t", aggregator=aggregator)
        self.assertEqual(result.steps[-1].name, "agg")
        self.assertEqual(result.output, "synthesized")
        # Aggregator saw both worker answers in its task.
        self.assertIn("alpha", result.steps[-1].task)
        self.assertIn("beta", result.steps[-1].task)

    def test_per_worker_tasks_for_map_reduce(self) -> None:
        workers = [
            make_fake_agent("w0", lambda task: f"out:{task}"),
            make_fake_agent("w1", lambda task: f"out:{task}"),
        ]
        result = run_parallel(workers, "ignored", tasks=["sub-a", "sub-b"])
        self.assertEqual(result.steps[0].output, "out:sub-a")
        self.assertEqual(result.steps[1].output, "out:sub-b")

    def test_tasks_length_mismatch_raises(self) -> None:
        workers = [make_fake_agent("w0", lambda task: "x")]
        with self.assertRaises(ValueError):
            run_parallel(workers, "t", tasks=["a", "b"])
