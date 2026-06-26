"""Goal loop unit tests (deterministic; no network)."""

from __future__ import annotations

import unittest

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import (
    AssistantMessage,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
)
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
        goal_events = [
            e for e in seen["state"].events if isinstance(e, GoalStatusEvent)
        ]
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

    def test_resume_step_output_ignores_previous_final(self):
        from simple_agent_lab.workflow import update_goal_tool

        tool = update_goal_tool()
        calls = {"n": 0}

        def generate(messages):
            del messages
            calls["n"] += 1
            if calls["n"] == 1:
                return AssistantMessage(
                    content=(TextBlock("first segment final"),),
                    sender="goal_agent",
                    target="user",
                    kind="final",
                )
            return AssistantMessage(
                content=(
                    TextBlock("second segment tool result"),
                    ToolCallBlock(
                        id="call_resume",
                        name="update_goal",
                        arguments={"status": "complete"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent(name="goal_agent", generate=generate, tools=(tool,))
        result = run_goal_loop(
            agent,
            "obj",
            check=_fails_n_then_passes(1),
            budgets=GoalBudgets(max_turns=2),
        )

        self.assertEqual(result.steps[0].output, "first segment final")
        self.assertEqual(result.steps[1].output, "second segment tool result")
        self.assertEqual(result.output, "second segment tool result")


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
            check=lambda s: CompletionResult(
                done=False, blocked=True, reason="no network"
            ),
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


class GoalLoopChecksTest(unittest.TestCase):
    def test_model_declared_completion_via_update_goal_tool(self):
        """Agent built with update_goal_tool(); generate closure calls it on
        the first (and only) inner turn → model_declared_check → complete."""
        from simple_agent_lab.workflow import (
            model_declared_check,
            update_goal_tool,
        )

        tool = update_goal_tool()
        call_count = {"n": 0}

        def generate(messages):
            call_count["n"] += 1
            # On first call emit the tool call; the loop terminates after it.
            return AssistantMessage(
                content=(
                    TextBlock("checking…"),
                    ToolCallBlock(
                        id="call_ug1",
                        name="update_goal",
                        arguments={"status": "complete", "reason": "all done"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent(name="goal_agent", generate=generate, tools=(tool,))
        result = run_goal_loop(
            agent,
            "do the thing",
            check=model_declared_check,
            budgets=GoalBudgets(max_turns=5),
        )
        self.assertEqual(result.status, "complete")

    def test_command_verifier_only_stops_when_command_passes(self):
        """command_verifier_check with a command that always exits 0 → complete."""
        from simple_agent_lab.workflow import command_verifier_check

        agent = _final_agent()
        # "true" is a shell builtin that always exits 0
        check = command_verifier_check("true")
        result = run_goal_loop(
            agent,
            "run something",
            check=check,
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "complete")

    def test_command_verifier_continues_when_command_fails(self):
        """command_verifier_check with a command that always exits nonzero → budget_exhausted."""
        from simple_agent_lab.workflow import command_verifier_check

        agent = _final_agent()
        # "false" is a shell builtin that always exits 1
        check = command_verifier_check("false")
        result = run_goal_loop(
            agent,
            "run something",
            check=check,
            budgets=GoalBudgets(max_turns=2),
        )
        self.assertEqual(result.status, "budget_exhausted")

    def _goal_agent(self, *, verify_command: str):
        """An agent that calls update_goal(complete, verify_command=...) each turn."""
        from simple_agent_lab.workflow import update_goal_tool

        def generate(messages):
            return AssistantMessage(
                content=(
                    ToolCallBlock(
                        id="call_ug",
                        name="update_goal",
                        arguments={
                            "status": "complete",
                            "reason": "claims done",
                            "verify_command": verify_command,
                        },
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        return Agent(name="goal_agent", generate=generate, tools=(update_goal_tool(),))

    def test_executed_check_completes_when_verify_command_passes(self):
        """A non-trivial verify_command that exits 0 → re-run passes → complete."""
        from simple_agent_lab.workflow import executed_completion_check

        result = run_goal_loop(
            self._goal_agent(verify_command="test 1 -eq 1"),
            "do the thing",
            check=executed_completion_check(),
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "complete")

    def test_executed_check_rejects_when_verify_command_fails(self):
        """A convincing claim whose verify_command exits nonzero never passes."""
        from simple_agent_lab.workflow import executed_completion_check

        result = run_goal_loop(
            self._goal_agent(verify_command="false"),
            "do the thing",
            check=executed_completion_check(),
            budgets=GoalBudgets(max_turns=2),
        )
        self.assertEqual(result.status, "budget_exhausted")

    def test_executed_check_rejects_empty_verify_command(self):
        """A declared completion with no verify_command is held open (an empty
        command would `run_bash("")` to exit 0 and pass the gate for free)."""
        from simple_agent_lab.workflow import executed_completion_check

        result = run_goal_loop(
            self._goal_agent(verify_command="  "),  # blank → empty after strip
            "do the thing",
            check=executed_completion_check(),
            budgets=GoalBudgets(max_turns=2),
        )
        self.assertEqual(result.status, "budget_exhausted")

    def test_default_check_continues_when_verifier_vetoes(self):
        """model declares done but verifier returns not-done → loop continues."""
        from simple_agent_lab.workflow import (
            default_check,
            update_goal_tool,
        )

        tool = update_goal_tool()

        def generate(messages):
            # Always call update_goal with status=complete
            return AssistantMessage(
                content=(
                    TextBlock("done!"),
                    ToolCallBlock(
                        id="call_dv1",
                        name="update_goal",
                        arguments={"status": "complete", "reason": "claimed done"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent(name="goal_agent", generate=generate, tools=(tool,))

        # Verifier always says not done → the default_check vetoes the model's claim
        def vetoing_verifier(s):
            return CompletionResult(done=False, reason="not really")

        result = run_goal_loop(
            agent,
            "do the thing",
            check=default_check(verifier=vetoing_verifier),
            budgets=GoalBudgets(max_turns=2),
        )
        # The verifier vetoes every claim → loop exhausts the budget
        self.assertEqual(result.status, "budget_exhausted")

    def _declaring_agent(self) -> Agent:
        """Agent that calls update_goal(complete) every turn."""
        from simple_agent_lab.workflow import update_goal_tool

        def generate(messages):
            return AssistantMessage(
                content=(
                    TextBlock("done"),
                    ToolCallBlock(
                        id="c",
                        name="update_goal",
                        arguments={"status": "complete", "reason": "x"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        return Agent(name="goal_agent", generate=generate, tools=(update_goal_tool(),))

    def _judge(self, verdict_json: str) -> Agent:
        def generate(messages):
            return AssistantMessage(
                content=(TextBlock(verdict_json),),
                sender="j",
                target="user",
                kind="final",
            )

        return Agent(name="j", generate=generate)

    def test_verified_completion_check_passes_when_judge_agrees(self):
        from simple_agent_lab.workflow import verified_completion_check

        result = run_goal_loop(
            self._declaring_agent(),
            "obj",
            check=verified_completion_check(self._judge('{"done": true}'), "obj"),
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "complete")

    def test_verified_completion_check_vetoes_when_judge_disagrees(self):
        from simple_agent_lab.workflow import verified_completion_check

        result = run_goal_loop(
            self._declaring_agent(),
            "obj",
            check=verified_completion_check(
                self._judge('{"done": false, "reason": "no tests run"}'), "obj"
            ),
            budgets=GoalBudgets(max_turns=2),
        )
        # Model declares done every turn but the judge keeps vetoing → budget out.
        self.assertEqual(result.status, "budget_exhausted")

    def test_judge_agent_check_with_stub_judge(self):
        """stub judge agent whose final message is '{"done": true}' → complete."""
        from simple_agent_lab.workflow import judge_agent_check

        def judge_generate(messages):
            return AssistantMessage(
                content=(TextBlock('{"done": true, "reason": "verified"}'),),
                sender="judge",
                target="user",
                kind="final",
            )

        judge = Agent(name="judge", generate=judge_generate)
        worker = _final_agent()
        check = judge_agent_check(judge, "some objective")
        result = run_goal_loop(
            worker,
            "some objective",
            check=check,
            budgets=GoalBudgets(max_turns=3),
        )
        self.assertEqual(result.status, "complete")

    def test_continuation_prompt_has_untrusted_wrapper_and_audit_text(self):
        """The continuation prompt contains '<untrusted_objective>' and 'NOT YET PROVEN'."""
        from simple_agent_lab.workflow.goal_loop import _continuation_prompt

        prompt = _continuation_prompt("write a thing")
        self.assertIn("<untrusted_objective>", prompt)
        self.assertIn("NOT YET PROVEN", prompt)
        self.assertIn("write a thing", prompt)

    def test_goal_prompt_has_untrusted_wrapper(self):
        """The first-turn prompt wraps the objective as untrusted data."""
        from simple_agent_lab.workflow.goal_loop import (
            _goal_prompt,
            UNTRUSTED_OBJECTIVE_PREAMBLE,
        )

        prompt = _goal_prompt("my objective")
        self.assertIn("<untrusted_objective>", prompt)
        self.assertIn("my objective", prompt)
        self.assertIn("untrusted_objective", UNTRUSTED_OBJECTIVE_PREAMBLE)

    def test_judge_agent_check_parse_failure_returns_not_done(self):
        """If the judge returns unparseable output, result is done=False."""
        from simple_agent_lab.workflow.goal_checks import _parse_judge_json

        result = _parse_judge_json("I cannot determine this")
        self.assertFalse(result.get("done"))

    def test_judge_agent_check_scalar_json_returns_not_done(self):
        """Valid non-object JSON still falls back instead of crashing."""
        from simple_agent_lab.workflow.goal_checks import _parse_judge_json

        result = _parse_judge_json("true")
        self.assertFalse(result.get("done"))
        self.assertEqual(result.get("reason"), "parse failure")

    def test_judge_agent_check_json_in_prose(self):
        """_parse_judge_json extracts JSON embedded in prose."""
        from simple_agent_lab.workflow.goal_checks import _parse_judge_json

        result = _parse_judge_json(
            'After analysis: {"done": true, "reason": "ok"} Done.'
        )
        self.assertTrue(result.get("done"))
        self.assertEqual(result.get("reason"), "ok")


class GoalLoopPromptInjectionTest(unittest.TestCase):
    """Custom initial/continuation prompt builders (the prompt-alignment seam)."""

    def _capturing_agent(self):
        from simple_agent_lab.messages import text_of

        seen: list[str] = []

        def generate(messages):
            # Record the latest task message this turn (resume re-shows the
            # original task, so only the most recent one is this turn's prompt).
            tasks = [m for m in messages if getattr(m, "kind", "") == "task"]
            if tasks:
                seen.append(text_of(tasks[-1].content))
            return AssistantMessage(
                content=(), sender="goal_agent", target="user", kind="final"
            )

        return Agent(name="goal_agent", generate=generate), seen

    def test_custom_builders_replace_default_wrapping(self):
        agent, seen = self._capturing_agent()
        run_goal_loop(
            agent,
            "RAW-OBJECTIVE",
            check=_fails_n_then_passes(1),  # one continuation, then done
            budgets=GoalBudgets(max_turns=5),
            initial_prompt=lambda o: o,
            continuation_prompt=lambda o: "CUSTOM-CONT",
        )
        # Turn 1 is the objective verbatim (no <untrusted_objective> wrapper);
        # the continuation is the caller's text.
        self.assertEqual(seen[0], "RAW-OBJECTIVE")
        self.assertEqual(seen[1], "CUSTOM-CONT")

    def test_default_builders_still_wrap_objective(self):
        agent, seen = self._capturing_agent()
        run_goal_loop(
            agent,
            "RAW-OBJECTIVE",
            check=_fails_n_then_passes(1),
            budgets=GoalBudgets(max_turns=5),
        )
        # Default path keeps the injection-safety wrapping for /goal objectives.
        self.assertIn("<untrusted_objective>", seen[0])
        self.assertIn("RAW-OBJECTIVE", seen[0])

    def test_model_declared_check_returns_blocked_on_blocked_status(self):
        """model_declared_check correctly maps status=blocked to CompletionResult(blocked=True)."""
        from simple_agent_lab.workflow import (
            model_declared_check,
            update_goal_tool,
        )

        tool = update_goal_tool()

        def generate(messages):
            return AssistantMessage(
                content=(
                    TextBlock("stuck"),
                    ToolCallBlock(
                        id="call_mb1",
                        name="update_goal",
                        arguments={"status": "blocked", "reason": "no network"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        agent = Agent(name="goal_agent", generate=generate, tools=(tool,))
        result = run_goal_loop(
            agent,
            "do a thing",
            check=model_declared_check,
            budgets=GoalBudgets(max_turns=5),
        )
        # same blocker repeated >= 3 times → blocked
        self.assertEqual(result.status, "blocked")


if __name__ == "__main__":
    unittest.main()
