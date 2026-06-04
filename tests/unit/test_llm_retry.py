"""Default LLM-layer retry (`simple_agent_lab.llm.retry`).

Retry for transient provider throttling lives in the LLM layer and is applied
once, by default, where the LLM-backed `generate` is built (`make_llm_agent`) —
so a long agent run survives a TPM/429 blip without each caller re-wrapping
`generate`. These cover the backoff mechanics, the throttling classifier, and
the wiring through `make_llm_agent`.
"""

from __future__ import annotations

import unittest
from typing import Iterator
from unittest import mock

from simple_agent_lab import make_llm_agent
from simple_agent_lab.llm import (
    LLMRequest,
    LLMResponse,
    LLMTool,
    Provider,
    StreamEvent,
    TextBlock,
    ToolCallBlock,
    complete_with_retry,
    complete_with_tool_call_retry,
    invalid_tool_call_reasons,
    is_retryable_llm_error,
    register_adapter,
)
from simple_agent_lab.llm.adapters import fake as fake_adapter
from simple_agent_lab.llm.types import RAW_ARGUMENTS_KEY
from simple_agent_lab.messages import message_tool_calls, text_of
from simple_agent_lab.tools.bash import make_bash_tool

_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def _request() -> LLMRequest:
    return LLMRequest(provider=_PROVIDER, messages=[])


class CompleteWithRetryTest(unittest.TestCase):
    def test_recovers_from_tpm_error_with_exponential_backoff(self) -> None:
        calls = 0
        sleeps: list[float] = []
        logs: list[str] = []
        expected = LLMResponse(content=(TextBlock("ok"),))

        def flaky_complete(request: LLMRequest) -> LLMResponse:
            nonlocal calls
            del request
            calls += 1
            if calls < 4:
                raise RuntimeError("TPM limit exceeded; retry after a while")
            return expected

        result = complete_with_retry(
            _request(),
            complete_fn=flaky_complete,
            sleep_fn=sleeps.append,
            log_fn=logs.append,
        )

        self.assertIs(result, expected)
        self.assertEqual(calls, 4)
        self.assertEqual(sleeps, [4.0, 8.0, 16.0])
        self.assertEqual(len(logs), 3)
        self.assertIn("attempt 1/20", logs[0])
        self.assertIn("retrying in 4s", logs[0])

    def test_caps_delay_at_sixty_seconds(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def flaky_complete(request: LLMRequest) -> LLMResponse:
            nonlocal calls
            del request
            calls += 1
            if calls < 8:
                raise RuntimeError("429 tokens per minute exceeded")
            return LLMResponse(content=(TextBlock("ok"),))

        complete_with_retry(
            _request(),
            complete_fn=flaky_complete,
            sleep_fn=sleeps.append,
            log_fn=lambda _: None,
        )

        self.assertEqual(sleeps, [4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0])

    def test_does_not_retry_non_throttling_errors(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def broken_complete(request: LLMRequest) -> LLMResponse:
            nonlocal calls
            del request
            calls += 1
            raise RuntimeError("invalid request body")

        with self.assertRaisesRegex(RuntimeError, "invalid request body"):
            complete_with_retry(
                _request(),
                complete_fn=broken_complete,
                sleep_fn=sleeps.append,
                log_fn=lambda _: None,
            )
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, [])

    def test_raises_last_error_after_twenty_attempts(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def failing_complete(request: LLMRequest) -> LLMResponse:
            nonlocal calls
            del request
            calls += 1
            raise RuntimeError(f"rate limit still active attempt={calls}")

        with self.assertRaisesRegex(RuntimeError, "attempt=20"):
            complete_with_retry(
                _request(),
                complete_fn=failing_complete,
                sleep_fn=sleeps.append,
                log_fn=lambda _: None,
            )
        self.assertEqual(calls, 20)
        self.assertEqual(len(sleeps), 19)
        self.assertEqual(sleeps[:5], [4.0, 8.0, 16.0, 32.0, 60.0])
        self.assertEqual(sleeps[-1], 60.0)


class IsRetryableLlmErrorTest(unittest.TestCase):
    def test_matches_common_tpm_and_rate_limit_text(self) -> None:
        self.assertTrue(is_retryable_llm_error(RuntimeError("TPM exceeded")))
        self.assertTrue(
            is_retryable_llm_error(RuntimeError("tokens per minute exhausted"))
        )
        self.assertTrue(is_retryable_llm_error(RuntimeError("HTTP 429")))
        self.assertTrue(is_retryable_llm_error(RuntimeError("Too Many Requests")))
        self.assertFalse(is_retryable_llm_error(RuntimeError("invalid schema")))


_BASH_TOOLS = [LLMTool(name="bash", description="run bash", parameters={})]


def _tool_request() -> LLMRequest:
    return LLMRequest(provider=_PROVIDER, messages=[], tools=_BASH_TOOLS)


def _tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=(ToolCallBlock(id="1", name=name, arguments=arguments),),
        stop_reason="tool_use",
    )


class InvalidToolCallReasonsTest(unittest.TestCase):
    def test_valid_call_has_no_reasons(self) -> None:
        response = _tool_call_response("bash", {"command": "ls"})
        self.assertEqual(invalid_tool_call_reasons(response, _BASH_TOOLS), [])

    def test_unknown_tool_is_flagged(self) -> None:
        response = _tool_call_response("ghost", {})
        reasons = invalid_tool_call_reasons(response, _BASH_TOOLS)
        self.assertEqual(len(reasons), 1)
        self.assertIn("ghost", reasons[0])

    def test_unparseable_arguments_are_flagged(self) -> None:
        response = _tool_call_response("bash", {RAW_ARGUMENTS_KEY: "{not json"})
        reasons = invalid_tool_call_reasons(response, _BASH_TOOLS)
        self.assertEqual(len(reasons), 1)
        self.assertIn("bash", reasons[0])

    def test_text_only_response_is_valid(self) -> None:
        response = LLMResponse(content=(TextBlock("hello"),))
        self.assertEqual(invalid_tool_call_reasons(response, _BASH_TOOLS), [])


class CompleteWithToolCallRetryTest(unittest.TestCase):
    def test_returns_first_valid_response_without_reasking(self) -> None:
        valid = _tool_call_response("bash", {"command": "ls"})
        calls: list[LLMRequest] = []

        def complete_fn(request: LLMRequest) -> LLMResponse:
            calls.append(request)
            return valid

        result = complete_with_tool_call_retry(
            _tool_request(), complete_fn=complete_fn, log_fn=lambda _: None
        )

        self.assertIs(result, valid)
        self.assertEqual(len(calls), 1)

    def test_reasks_until_valid_with_corrective_runtime_message(self) -> None:
        bad = _tool_call_response("ghost", {})
        good = _tool_call_response("bash", {"command": "ls"})
        responses = iter([bad, good])
        seen: list[LLMRequest] = []

        def complete_fn(request: LLMRequest) -> LLMResponse:
            seen.append(request)
            return next(responses)

        result = complete_with_tool_call_retry(
            _tool_request(), complete_fn=complete_fn, log_fn=lambda _: None
        )

        self.assertIs(result, good)
        self.assertEqual(len(seen), 2)
        # The original request is untouched; the re-ask carries one extra
        # system-role nudge naming the offending tool and the valid tools.
        self.assertEqual(seen[0].messages, [])
        self.assertEqual(len(seen[1].messages), 1)
        nudge = seen[1].messages[0]
        self.assertEqual(nudge.role, "system")
        self.assertIn("ghost", text_of(nudge.content))
        self.assertIn("bash", text_of(nudge.content))

    def test_degrades_to_last_response_after_max_attempts(self) -> None:
        bad = _tool_call_response("ghost", {})
        calls: list[LLMRequest] = []

        def complete_fn(request: LLMRequest) -> LLMResponse:
            calls.append(request)
            return bad

        result = complete_with_tool_call_retry(
            _tool_request(), complete_fn=complete_fn, log_fn=lambda _: None
        )

        self.assertIs(result, bad)
        self.assertEqual(len(calls), 3)  # initial + 2 re-asks


class MakeLlmAgentRetryWiringTest(unittest.TestCase):
    """`make_llm_agent` retries both hiccups by default — no per-caller wrapper."""

    def tearDown(self) -> None:
        # The fake adapter is process-global; restore it after overriding.
        register_adapter("fake", fake_adapter.stream)

    def test_generate_retries_transient_throttling_by_default(self) -> None:
        attempts = 0

        def flaky(req: LLMRequest) -> Iterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("429 Too Many Requests")
            response = LLMResponse(
                content=(TextBlock("done"),),
                stop_reason="end_turn",
                model=req.provider.model,
            )
            yield StreamEvent(kind="done", payload={"response": response})

        register_adapter("fake", flaky)
        agent = make_llm_agent(name="t", provider=_PROVIDER)

        with mock.patch("simple_agent_lab.llm.retry.time.sleep") as sleep:
            output = agent.generate([])

        self.assertEqual(attempts, 2)
        self.assertEqual(output.kind, "final")
        sleep.assert_called_once()

    def test_generate_repairs_invalid_tool_call_by_default(self) -> None:
        attempts = 0

        def flaky(req: LLMRequest) -> Iterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                call = ToolCallBlock(id="1", name="ghost_tool", arguments={})
            else:
                call = ToolCallBlock(
                    id="2",
                    name="bash",
                    arguments={"command": "ls", "description": "list files"},
                )
            response = LLMResponse(
                content=(call,), stop_reason="tool_use", model=req.provider.model
            )
            yield StreamEvent(kind="done", payload={"response": response})

        register_adapter("fake", flaky)
        agent = make_llm_agent(name="t", provider=_PROVIDER, tools=[make_bash_tool()])

        output = agent.generate([])

        self.assertEqual(attempts, 2)
        self.assertEqual([call.name for call in message_tool_calls(output)], ["bash"])


if __name__ == "__main__":
    unittest.main()
