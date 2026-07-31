"""Goal loop unit tests (deterministic; no network)."""

from __future__ import annotations

import unittest

from simple_long_horizon_agent.core import Agent
from simple_long_horizon_agent.messages import (
    AssistantMessage,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    text_of,
)
from simple_long_horizon_agent.protocols import GoalStatusEvent
from simple_long_horizon_agent.workflow import (
    CompletionResult,
    GoalBudgets,
    command_verifier_check,
    default_check,
    executed_completion_check,
    judge_agent_check,
    model_declared_check,
    run_goal_loop,
    update_goal_tool,
    verified_completion_check,
)
from simple_long_horizon_agent.workflow.goal_checks import _parse_judge_json
from simple_long_horizon_agent.workflow.goal_loop import (
    UNTRUSTED_OBJECTIVE_PREAMBLE,
    _continuation_prompt,
    _goal_prompt,
)


def _final_agent(
    *, name: str = "goal_agent", output: str = "", output_tokens: int = 0
) -> Agent:
    content = (TextBlock(output),) if output else ()
    usage = TokenUsage(output_tokens=output_tokens) if output_tokens else None

    def generate(messages):
        return AssistantMessage(
            content=content,
            sender=name,
            target="user",
            kind="final",
            usage=usage,
        )

    return Agent(name=name, generate=generate)


def _fails_n_then_passes(n: int):
    calls = 0

    def check(state):
        nonlocal calls
        done = calls >= n
        calls += 1
        return CompletionResult(done=done)

    return check


def _run(
    check,
    *,
    agent: Agent | None = None,
    objective: str = "obj",
    max_turns: int | None = None,
    budgets: GoalBudgets | None = None,
    **kwargs,
):
    if budgets is None:
        budgets = GoalBudgets(max_turns=max_turns)
    return run_goal_loop(
        agent or _final_agent(),
        objective,
        check=check,
        budgets=budgets,
        **kwargs,
    )


def _declaring_agent(
    *,
    status: str = "complete",
    reason: str = "claimed done",
    verify_command: str | None = None,
) -> Agent:
    arguments = {"status": status, "reason": reason}
    if verify_command is not None:
        arguments["verify_command"] = verify_command

    def generate(messages):
        return AssistantMessage(
            content=(
                TextBlock("done"),
                ToolCallBlock(
                    id="update_goal_call",
                    name="update_goal",
                    arguments=arguments,
                ),
            ),
            sender="goal_agent",
            target="goal_agent",
            kind="step",
        )

    return Agent(
        name="goal_agent",
        generate=generate,
        tools=(update_goal_tool(),),
    )


def _judge(verdict: str) -> Agent:
    return _final_agent(name="judge", output=verdict)


def _capturing_agent() -> tuple[Agent, list[str]]:
    seen: list[str] = []

    def generate(messages):
        tasks = [message for message in messages if message.kind == "task"]
        if tasks:
            seen.append(text_of(tasks[-1].content))
        return AssistantMessage(
            content=(),
            sender="goal_agent",
            target="user",
            kind="final",
        )

    return Agent(name="goal_agent", generate=generate), seen


class GoalLoopLifecycleTest(unittest.TestCase):
    def test_continuations_report_completion_turns_and_steps(self):
        result = _run(_fails_n_then_passes(2), max_turns=10)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.turns_used, 2)
        self.assertEqual(len(result.steps), 3)

    def test_turn_budget_reports_exact_exhaustion(self):
        result = _run(
            lambda state: CompletionResult(done=False),
            max_turns=3,
        )

        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.turns_used, 3)

    def test_goal_events_are_replayable_and_not_scratch_data(self):
        seen = []

        def check(state):
            seen.append(state)
            return CompletionResult(done=True)

        _run(check, objective="record me")
        state = seen[-1]
        events = [event for event in state.events if isinstance(event, GoalStatusEvent)]

        self.assertEqual([event.status for event in events], ["active", "complete"])
        self.assertTrue(all(event.objective == "record me" for event in events))
        self.assertNotIn("goal", state.data)

        before = [event.status for event in events]
        state.rebuild_snapshot()
        after = [
            event.status for event in state.events if isinstance(event, GoalStatusEvent)
        ]
        self.assertEqual(after, before)

    def test_resume_output_uses_only_the_latest_segment(self):
        calls = 0

        def generate(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
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
                        id="resume_call",
                        name="update_goal",
                        arguments={"status": "complete"},
                    ),
                ),
                sender="goal_agent",
                target="goal_agent",
                kind="step",
            )

        result = _run(
            _fails_n_then_passes(1),
            agent=Agent(
                name="goal_agent",
                generate=generate,
                tools=(update_goal_tool(),),
            ),
            max_turns=2,
        )

        self.assertEqual(
            [step.output for step in result.steps],
            ["first segment final", "second segment tool result"],
        )
        self.assertEqual(result.output, "second segment tool result")

    def test_token_and_abort_budgets(self):
        def never_done(state):
            return CompletionResult(done=False)

        token_result = _run(
            never_done,
            agent=_final_agent(output_tokens=50),
            budgets=GoalBudgets(token_budget=120),
        )
        self.assertEqual(token_result.status, "budget_exhausted")
        self.assertGreaterEqual(token_result.tokens_used, 120)

        for name, kwargs in (
            ("caller", {"abort": lambda: True}),
            ("wall clock", {"budgets": GoalBudgets(wall_clock_seconds=0.0)}),
        ):
            with self.subTest(abort_source=name):
                self.assertEqual(_run(never_done, **kwargs).status, "aborted")

    def test_token_usage_is_reported_in_result_and_events(self):
        seen = []
        verdict = _fails_n_then_passes(1)

        def check(state):
            seen.append(state)
            return verdict(state)

        result = _run(
            check,
            agent=_final_agent(output_tokens=30),
            max_turns=5,
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.tokens_used, 60)
        events = [
            event for event in seen[-1].events if isinstance(event, GoalStatusEvent)
        ]
        self.assertTrue(all(event.tokens_used >= 30 for event in events))

    def test_blocking_requires_three_consecutive_matching_reasons(self):
        def changing_reasons():
            reasons = iter(("a", "b", "c", "d"))
            return lambda state: CompletionResult(
                done=False,
                blocked=True,
                reason=next(reasons),
            )

        def alternating_blocked():
            calls = 0

            def check(state):
                nonlocal calls
                blocked = calls % 2 == 0
                calls += 1
                return CompletionResult(
                    done=False,
                    blocked=blocked,
                    reason="same blocker" if blocked else "",
                )

            return check

        cases = (
            (
                "same reason reaches threshold",
                lambda state: CompletionResult(
                    done=False,
                    blocked=True,
                    reason="no network",
                ),
                10,
                "blocked",
            ),
            (
                "changing reason breaks streak",
                changing_reasons(),
                3,
                "budget_exhausted",
            ),
            (
                "non-blocked verdict resets streak",
                alternating_blocked(),
                4,
                "budget_exhausted",
            ),
        )
        for name, check, max_turns, expected in cases:
            with self.subTest(policy=name):
                self.assertEqual(
                    _run(check, max_turns=max_turns).status,
                    expected,
                )


class GoalLoopChecksTest(unittest.TestCase):
    def test_model_declared_terminal_statuses(self):
        for declared, expected in (
            ("complete", "complete"),
            ("blocked", "blocked"),
        ):
            with self.subTest(declared=declared):
                result = _run(
                    model_declared_check,
                    agent=_declaring_agent(
                        status=declared,
                        reason="no network" if declared == "blocked" else "all done",
                    ),
                    max_turns=5,
                )
                self.assertEqual(result.status, expected)

    def test_command_verifier_accepts_zero_and_rejects_nonzero(self):
        for command, expected in (
            ("true", "complete"),
            ("false", "budget_exhausted"),
        ):
            with self.subTest(command=command):
                result = _run(command_verifier_check(command), max_turns=2)
                self.assertEqual(result.status, expected)

    def test_executed_verifier_requires_a_passing_nonempty_command(self):
        for command, expected in (
            ("test 1 -eq 1", "complete"),
            ("false", "budget_exhausted"),
            ("  ", "budget_exhausted"),
        ):
            with self.subTest(verify_command=repr(command)):
                result = _run(
                    executed_completion_check(),
                    agent=_declaring_agent(verify_command=command),
                    max_turns=2,
                )
                self.assertEqual(result.status, expected)

    def test_default_check_honors_verifier_veto(self):
        def veto(state):
            return CompletionResult(done=False, reason="not really")

        result = _run(
            default_check(verifier=veto),
            agent=_declaring_agent(),
            max_turns=2,
        )
        self.assertEqual(result.status, "budget_exhausted")

    def test_verified_completion_honors_judge_verdict(self):
        for verdict, expected in (
            ('{"done": true}', "complete"),
            ('{"done": false, "reason": "no tests run"}', "budget_exhausted"),
        ):
            with self.subTest(verdict=verdict):
                result = _run(
                    verified_completion_check(_judge(verdict), "obj"),
                    agent=_declaring_agent(),
                    max_turns=2,
                )
                self.assertEqual(result.status, expected)

    def test_judge_agent_check_accepts_stub_judge(self):
        result = _run(
            judge_agent_check(
                _judge('{"done": true, "reason": "verified"}'),
                "some objective",
            ),
            objective="some objective",
            max_turns=3,
        )
        self.assertEqual(result.status, "complete")

    def test_judge_json_parsing_shapes(self):
        cases = (
            ("unparseable", "I cannot determine this", False, "parse failure"),
            ("scalar JSON", "true", False, "parse failure"),
            (
                "JSON in prose",
                'After analysis: {"done": true, "reason": "ok"} Done.',
                True,
                "ok",
            ),
        )
        for name, payload, expected_done, expected_reason in cases:
            with self.subTest(shape=name):
                verdict = _parse_judge_json(payload)
                self.assertEqual(verdict.get("done"), expected_done)
                self.assertEqual(verdict.get("reason"), expected_reason)


class GoalLoopPromptTest(unittest.TestCase):
    def test_default_prompts_wrap_untrusted_objective_and_audit_completion(self):
        cases = (
            ("initial", _goal_prompt("my objective"), ()),
            ("continuation", _continuation_prompt("my objective"), ("NOT YET PROVEN",)),
        )
        for name, prompt, extra_fragments in cases:
            with self.subTest(prompt=name):
                for fragment in (
                    "<untrusted_objective>",
                    "my objective",
                    *extra_fragments,
                ):
                    self.assertIn(fragment, prompt)
        self.assertIn("untrusted_objective", UNTRUSTED_OBJECTIVE_PREAMBLE)

    def test_custom_builders_replace_defaults_while_defaults_remain_wrapped(self):
        custom_agent, custom_seen = _capturing_agent()
        _run(
            _fails_n_then_passes(1),
            agent=custom_agent,
            objective="RAW-OBJECTIVE",
            max_turns=5,
            initial_prompt=lambda objective: objective,
            continuation_prompt=lambda objective: "CUSTOM-CONT",
        )
        self.assertEqual(custom_seen, ["RAW-OBJECTIVE", "CUSTOM-CONT"])

        default_agent, default_seen = _capturing_agent()
        _run(
            _fails_n_then_passes(1),
            agent=default_agent,
            objective="RAW-OBJECTIVE",
            max_turns=5,
        )
        self.assertIn("<untrusted_objective>", default_seen[0])
        self.assertIn("RAW-OBJECTIVE", default_seen[0])
