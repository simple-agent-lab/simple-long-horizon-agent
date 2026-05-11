"""Unit tests for the real-provider LLM adapters.

These tests do NOT require `anthropic` or `openai` to be installed: they
stub the SDK modules in `sys.modules` so each adapter sees a fake client
whose call records the kwargs and returns a canned response. That lets us
verify wire-shape translation (message bridging, tool schemas, tool-result
roundtrip, usage mapping, stop-reason mapping) in pure Python.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

from simple_agent_lab.llm import (
    ContentBlock,
    LLMMessage,
    LLMRequest,
    LLMTool,
    Provider,
    StreamEvent,
    ToolCall,
    complete,
    iter_stream,
)


ANTHROPIC_PROVIDER = Provider(
    id="claude-test",
    api="anthropic-messages",
    model="claude-test-1",
    api_key_env="TEST_ANTHROPIC_KEY",
)

OPENAI_CHAT_PROVIDER = Provider(
    id="gpt-chat-test",
    api="openai-chat",
    model="gpt-test-1",
    api_key_env="TEST_OPENAI_KEY",
)

OPENAI_RESPONSES_PROVIDER = Provider(
    id="gpt-resp-test",
    api="openai-responses",
    model="gpt-test-1",
    api_key_env="TEST_OPENAI_KEY",
)


def _bash_tool() -> LLMTool:
    return LLMTool(
        name="bash",
        description="Run a bash command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )


def _tool_use_request() -> LLMRequest:
    """A request that includes a prior tool_use + tool_result roundtrip."""
    return LLMRequest(
        provider=ANTHROPIC_PROVIDER,
        system_prompt="You are a tiny bash agent.",
        messages=[
            LLMMessage(role="user", content="Run: echo hello"),
            LLMMessage(
                role="assistant",
                content="Calling bash.",
                tool_calls=[ToolCall(id="t1", name="bash", arguments={"command": "echo hello"})],
            ),
            LLMMessage(role="tool_result", content="hello", tool_call_id="t1", name="bash"),
        ],
        tools=[_bash_tool()],
    )


@contextmanager
def _stub_module(name: str, module: types.ModuleType):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _capture_kwargs() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def _stub_anthropic(response: Any, captured: dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType("anthropic")

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    class Anthropic:
        def __init__(self, *, api_key=None, base_url=None):
            captured["__init_api_key"] = api_key
            captured["__init_base_url"] = base_url
            self.messages = FakeMessages()

    module.Anthropic = Anthropic
    return module


def _anthropic_response(
    *,
    text: str = "ok",
    tool_uses: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
) -> Any:
    content: list[Any] = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    for tu in tool_uses or []:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id=tu["id"],
                name=tu["name"],
                input=tu.get("input", {}),
            )
        )
    return SimpleNamespace(
        id="msg_123",
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


class AnthropicAdapterTest(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_anthropic(_anthropic_response(), captured)
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        with _stub_module("anthropic", module), mock.patch.dict("os.environ", {}, clear=False):
            # Make sure the env var is unset.
            import os

            os.environ.pop("TEST_ANTHROPIC_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "TEST_ANTHROPIC_KEY"):
                complete(req)

    def test_builds_request_and_emits_events(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_anthropic(
            _anthropic_response(
                text="Final answer.",
                tool_uses=[{"id": "t2", "name": "bash", "input": {"command": "ls"}}],
                stop_reason="tool_use",
            ),
            captured,
        )
        req = _tool_use_request()
        events: list[StreamEvent] = []
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}, clear=False),
        ):
            events = list(iter_stream(req))

        self.assertEqual(captured["__init_api_key"], "k")
        self.assertEqual(captured["model"], "claude-test-1")
        self.assertEqual(captured["system"], "You are a tiny bash agent.")
        self.assertEqual(
            captured["tools"],
            [
                {
                    "name": "bash",
                    "description": "Run a bash command.",
                    "input_schema": _bash_tool().parameters,
                }
            ],
        )
        # max_tokens default applied.
        self.assertIn("max_tokens", captured)

        messages = captured["messages"]
        self.assertEqual(messages[0], {"role": "user", "content": "Run: echo hello"})
        self.assertEqual(messages[1]["role"], "assistant")
        assistant_blocks = messages[1]["content"]
        self.assertEqual(assistant_blocks[0], {"type": "text", "text": "Calling bash."})
        self.assertEqual(
            assistant_blocks[1],
            {
                "type": "tool_use",
                "id": "t1",
                "name": "bash",
                "input": {"command": "echo hello"},
            },
        )
        # tool_result is wrapped as a user message with a tool_result block.
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(
            messages[2]["content"][0],
            {"type": "tool_result", "tool_use_id": "t1", "content": "hello"},
        )

        kinds = [event.kind for event in events]
        self.assertEqual(
            kinds,
            [
                "text_delta",
                "tool_call_start",
                "tool_call_complete",
                "usage_update",
                "done",
            ],
        )
        done = events[-1]
        response = done.payload["response"]
        self.assertEqual(response.text, "Final answer.")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "t2")
        self.assertEqual(response.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(response.usage.input_tokens, 11)
        self.assertEqual(response.usage.output_tokens, 5)

    def test_drops_assistant_message_with_no_blocks(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_anthropic(_anthropic_response(text="hi"), captured)
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(role="user", content="hi"),
                LLMMessage(role="assistant", content=""),  # empty, should drop
            ],
        )
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}, clear=False),
        ):
            complete(req)
        # Only the original user message should be sent.
        self.assertEqual(len(captured["messages"]), 1)


# ---------------------------------------------------------------------------
# OpenAI Chat
# ---------------------------------------------------------------------------


def _stub_openai(response: Any, captured: dict[str, Any], *, kind: str) -> types.ModuleType:
    module = types.ModuleType("openai")

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    class OpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            captured["__init_api_key"] = api_key
            captured["__init_base_url"] = base_url
            if kind == "chat":
                self.chat = FakeChat()
            else:
                self.responses = FakeResponses()

    module.OpenAI = OpenAI
    return module


def _chat_response(
    *,
    text: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 13,
    completion_tokens: int = 4,
    cached: int = 0,
    reasoning_content: str | None = None,
) -> Any:
    tcs = []
    for tc in tool_calls or []:
        tcs.append(
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
        )
    message_kwargs: dict[str, Any] = {"content": text, "tool_calls": tcs or None}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content
    message = SimpleNamespace(**message_kwargs)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached) if cached else None,
    )
    return SimpleNamespace(id="cmpl_1", choices=[choice], usage=usage)


class OpenAIChatAdapterTest(unittest.TestCase):
    def test_builds_request_and_emits_events(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(
            _chat_response(
                text="Calling bash.",
                tool_calls=[
                    {"id": "t2", "name": "bash", "arguments": json.dumps({"command": "ls"})}
                ],
                finish_reason="tool_calls",
            ),
            captured,
            kind="chat",
        )
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            system_prompt="be helpful",
            messages=[
                LLMMessage(role="user", content="run ls"),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="t1", name="bash", arguments={"command": "echo hello"})
                    ],
                ),
                LLMMessage(
                    role="tool_result",
                    content="hello",
                    tool_call_id="t1",
                    name="bash",
                ),
            ],
            tools=[_bash_tool()],
        )
        events: list[StreamEvent] = []
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            events = list(iter_stream(req))

        self.assertEqual(captured["__init_api_key"], "k")
        self.assertEqual(captured["model"], "gpt-test-1")

        messages = captured["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "be helpful"})
        self.assertEqual(messages[1], {"role": "user", "content": "run ls"})
        assistant_entry = messages[2]
        self.assertEqual(assistant_entry["role"], "assistant")
        self.assertIsNone(assistant_entry["content"])
        self.assertEqual(len(assistant_entry["tool_calls"]), 1)
        self.assertEqual(assistant_entry["tool_calls"][0]["id"], "t1")
        self.assertEqual(assistant_entry["tool_calls"][0]["type"], "function")
        self.assertEqual(
            assistant_entry["tool_calls"][0]["function"]["name"], "bash"
        )
        self.assertEqual(
            json.loads(assistant_entry["tool_calls"][0]["function"]["arguments"]),
            {"command": "echo hello"},
        )
        self.assertEqual(
            messages[3],
            {"role": "tool", "tool_call_id": "t1", "content": "hello"},
        )

        self.assertEqual(
            captured["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a bash command.",
                        "parameters": _bash_tool().parameters,
                    },
                }
            ],
        )

        kinds = [event.kind for event in events]
        self.assertEqual(
            kinds,
            [
                "text_delta",
                "tool_call_start",
                "tool_call_complete",
                "usage_update",
                "done",
            ],
        )
        response = events[-1].payload["response"]
        self.assertEqual(response.text, "Calling bash.")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(response.usage.input_tokens, 13)
        self.assertEqual(response.usage.output_tokens, 4)

    def test_handles_response_without_text_or_tools(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(
            _chat_response(text=None, finish_reason="length"),
            captured,
            kind="chat",
        )
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            events = list(iter_stream(req))
        kinds = [event.kind for event in events]
        # No text → no text_delta event.
        self.assertEqual(kinds, ["usage_update", "done"])
        response = events[-1].payload["response"]
        self.assertEqual(response.text, "")
        self.assertEqual(response.stop_reason, "max_tokens")

    def test_missing_api_key_raises_clear_error(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(_chat_response(), captured, kind="chat")
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        with _stub_module("openai", module):
            import os

            os.environ.pop("TEST_OPENAI_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "TEST_OPENAI_KEY"):
                complete(req)


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def _responses_response(
    *,
    text_blocks: list[str] | None = None,
    function_calls: list[dict[str, Any]] | None = None,
    incomplete_reason: str | None = None,
    status: str = "completed",
    input_tokens: int = 7,
    output_tokens: int = 3,
    cached: int = 0,
) -> Any:
    output_items: list[Any] = []
    if text_blocks:
        content = [
            SimpleNamespace(type="output_text", text=block) for block in text_blocks
        ]
        output_items.append(
            SimpleNamespace(type="message", role="assistant", content=content)
        )
    for fc in function_calls or []:
        output_items.append(
            SimpleNamespace(
                type="function_call",
                call_id=fc["call_id"],
                name=fc["name"],
                arguments=fc["arguments"],
            )
        )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached) if cached else None,
    )
    incomplete_details = (
        SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
    )
    return SimpleNamespace(
        id="resp_1",
        output=output_items,
        usage=usage,
        status=status,
        incomplete_details=incomplete_details,
    )


class OpenAIResponsesAdapterTest(unittest.TestCase):
    def test_builds_request_with_instructions_and_flat_tools(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(
            _responses_response(
                text_blocks=["First.", " Second."],
                function_calls=[
                    {
                        "call_id": "call_42",
                        "name": "bash",
                        "arguments": json.dumps({"command": "pwd"}),
                    }
                ],
            ),
            captured,
            kind="responses",
        )
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            system_prompt="be brief",
            messages=[
                LLMMessage(role="user", content="hi"),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="prev", name="bash", arguments={"command": "ls"})
                    ],
                ),
                LLMMessage(
                    role="tool_result",
                    content="out",
                    tool_call_id="prev",
                    name="bash",
                ),
            ],
            tools=[_bash_tool()],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            events = list(iter_stream(req))

        self.assertEqual(captured["instructions"], "be brief")
        self.assertEqual(
            captured["tools"],
            [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "Run a bash command.",
                    "parameters": _bash_tool().parameters,
                }
            ],
        )
        input_items = captured["input"]
        # user message → input_text content item.
        self.assertEqual(input_items[0]["type"], "message")
        self.assertEqual(input_items[0]["role"], "user")
        self.assertEqual(
            input_items[0]["content"], [{"type": "input_text", "text": "hi"}]
        )
        # assistant w/ no text but tool_call → function_call item only.
        self.assertEqual(input_items[1]["type"], "function_call")
        self.assertEqual(input_items[1]["call_id"], "prev")
        self.assertEqual(
            json.loads(input_items[1]["arguments"]), {"command": "ls"}
        )
        # tool_result → function_call_output item.
        self.assertEqual(input_items[2]["type"], "function_call_output")
        self.assertEqual(input_items[2]["call_id"], "prev")
        self.assertEqual(input_items[2]["output"], "out")

        response = events[-1].payload["response"]
        self.assertEqual(response.text, "First. Second.")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "call_42")
        self.assertEqual(response.tool_calls[0].arguments, {"command": "pwd"})
        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 3)

    def test_incomplete_max_tokens_maps_to_max_tokens_stop(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(
            _responses_response(
                text_blocks=["partial"],
                incomplete_reason="max_output_tokens",
                status="incomplete",
            ),
            captured,
            kind="responses",
        )
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            response = complete(req)
        self.assertEqual(response.stop_reason, "max_tokens")


# ---------------------------------------------------------------------------
# Reasoning continuity across multi-turn tool use
#
# These tests cover the contract that makes reasoning a first-class citizen
# of the trajectory:
#   1. Inbound:  the adapter lifts the model's reasoning into a thinking
#                ContentBlock on LLMResponse.content (so the response carries
#                reasoning alongside text and tool_calls in stable order).
#   2. Outbound: when an assistant message carries thinking blocks, the next
#                request replays them in whatever wire shape the provider
#                expects — `reasoning_content` field for OpenAI-compat,
#                `{"type": "thinking", ...}` blocks for Anthropic.
#   3. Opt-out:  flipping Provider.replay_reasoning=False suppresses the
#                outbound side for endpoints that reject the replay shape.
# ---------------------------------------------------------------------------


class OpenAIChatReasoningTest(unittest.TestCase):
    def test_inbound_reasoning_content_becomes_thinking_block(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(
            _chat_response(
                text="Result: 6.",
                reasoning_content="2*3 is 6.",
                finish_reason="stop",
            ),
            captured,
            kind="chat",
        )
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            messages=[LLMMessage(role="user", content="2*3?")],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            events = list(iter_stream(req))

        kinds = [event.kind for event in events]
        self.assertIn("thinking_delta", kinds)
        # Thinking arrives before text so consumers see the model's
        # reasoning step before the user-visible answer.
        self.assertLess(kinds.index("thinking_delta"), kinds.index("text_delta"))

        response = events[-1].payload["response"]
        thinking_blocks = response.thinking_blocks
        self.assertEqual(len(thinking_blocks), 1)
        self.assertEqual(thinking_blocks[0].thinking, "2*3 is 6.")
        # Derived views agree with content.
        self.assertEqual(response.thinking, "2*3 is 6.")
        self.assertEqual(response.text, "Result: 6.")
        # Content preserves order: thinking, then text.
        self.assertEqual([block.kind for block in response.content], ["thinking", "text"])

    def test_outbound_replays_thinking_as_reasoning_content(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(_chat_response(text="ok"), captured, kind="chat")
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,  # replay_reasoning=True by default
            messages=[
                LLMMessage(role="user", content="Compute 2*3."),
                LLMMessage(
                    role="assistant",
                    content=[
                        ContentBlock(kind="thinking", thinking="2*3 is 6."),
                        ContentBlock(kind="text", text="6"),
                    ],
                ),
                LLMMessage(role="user", content="now double it"),
            ],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            complete(req)

        assistant_entry = next(m for m in captured["messages"] if m["role"] == "assistant")
        self.assertEqual(assistant_entry["reasoning_content"], "2*3 is 6.")
        self.assertEqual(assistant_entry["content"], "6")

    def test_outbound_skips_replay_when_opted_out(self) -> None:
        captured: dict[str, Any] = {}
        module = _stub_openai(_chat_response(text="ok"), captured, kind="chat")
        provider = Provider(
            id="gpt-chat-test",
            api="openai-chat",
            model="gpt-test-1",
            api_key_env="TEST_OPENAI_KEY",
            replay_reasoning=False,
        )
        req = LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="hi"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ContentBlock(kind="thinking", thinking="careful now"),
                        ContentBlock(kind="text", text="hello"),
                    ],
                ),
                LLMMessage(role="user", content="again"),
            ],
        )
        with (
            _stub_module("openai", module),
            mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}, clear=False),
        ):
            complete(req)

        assistant_entry = next(m for m in captured["messages"] if m["role"] == "assistant")
        self.assertNotIn("reasoning_content", assistant_entry)


class AnthropicReasoningReplayTest(unittest.TestCase):
    def _anthropic_response_with_thinking(self, captured: dict[str, Any]) -> types.ModuleType:
        module = types.ModuleType("anthropic")

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    id="msg_1",
                    content=[
                        SimpleNamespace(
                            type="thinking", thinking="careful step", signature="sig-1"
                        ),
                        SimpleNamespace(type="text", text="done"),
                    ],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=1,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                )

        class Anthropic:
            def __init__(self, *, api_key=None, base_url=None):
                self.messages = FakeMessages()

        module.Anthropic = Anthropic
        return module

    def test_inbound_captures_thinking_with_signature(self) -> None:
        captured: dict[str, Any] = {}
        module = self._anthropic_response_with_thinking(captured)
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}, clear=False),
        ):
            response = complete(req)

        self.assertEqual(len(response.thinking_blocks), 1)
        block = response.thinking_blocks[0]
        self.assertEqual(block.thinking, "careful step")
        self.assertEqual(block.signature, "sig-1")
        self.assertFalse(block.redacted)
        # Content order matches the wire order Anthropic returned.
        self.assertEqual([b.kind for b in response.content], ["thinking", "text"])

    def test_outbound_replays_thinking_block_first(self) -> None:
        captured: dict[str, Any] = {}
        module = self._anthropic_response_with_thinking(captured)
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(role="user", content="go"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ContentBlock(
                            kind="thinking",
                            thinking="careful step",
                            signature="sig-1",
                        ),
                        ContentBlock(kind="text", text="done"),
                    ],
                    tool_calls=[ToolCall(id="t1", name="bash", arguments={"command": "ls"})],
                ),
                LLMMessage(role="tool_result", content="out", tool_call_id="t1", name="bash"),
            ],
        )
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}, clear=False),
        ):
            complete(req)

        assistant_msg = next(m for m in captured["messages"] if m["role"] == "assistant")
        blocks = assistant_msg["content"]
        # Anthropic requires thinking to lead text and tool_use; verify order.
        self.assertEqual(blocks[0]["type"], "thinking")
        self.assertEqual(blocks[0]["thinking"], "careful step")
        self.assertEqual(blocks[0]["signature"], "sig-1")
        self.assertEqual(blocks[1]["type"], "text")
        self.assertEqual(blocks[2]["type"], "tool_use")

    def test_outbound_skips_replay_when_opted_out(self) -> None:
        captured: dict[str, Any] = {}
        module = self._anthropic_response_with_thinking(captured)
        provider = Provider(
            id="claude-test",
            api="anthropic-messages",
            model="claude-test-1",
            api_key_env="TEST_ANTHROPIC_KEY",
            replay_reasoning=False,
        )
        req = LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="go"),
                LLMMessage(
                    role="assistant",
                    content=[ContentBlock(kind="thinking", thinking="x", signature="s")],
                ),
                LLMMessage(role="user", content="again"),
            ],
        )
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}, clear=False),
        ):
            complete(req)

        assistant_messages = [m for m in captured["messages"] if m["role"] == "assistant"]
        # No assistant message should slip through carrying a thinking block.
        for msg in assistant_messages:
            for block in msg["content"]:
                self.assertNotEqual(block["type"], "thinking")


class BridgeThinkingPreservationTest(unittest.TestCase):
    def test_thinking_blocks_carry_signature_into_assistant_message(self) -> None:
        from simple_agent_lab.llm.bridge import llm_response_to_assistant_message
        from simple_agent_lab.llm.types import LLMResponse
        from simple_agent_lab.messages import AssistantMessage

        response = LLMResponse(
            content=[
                ContentBlock(kind="thinking", thinking="step 1", signature="s1"),
                ContentBlock(kind="thinking", thinking="step 2", signature="s2", redacted=True),
                ContentBlock(kind="text", text="answer"),
            ],
        )
        msg = llm_response_to_assistant_message(
            response, sender="agent", target="user", kind="final"
        )
        assert isinstance(msg, AssistantMessage)
        self.assertEqual(len(msg.thinking), 2)
        self.assertEqual(msg.thinking[0].text, "step 1")
        self.assertEqual(msg.thinking[0].signature, "s1")
        self.assertFalse(msg.thinking[0].redacted)
        self.assertEqual(msg.thinking[1].signature, "s2")
        self.assertTrue(msg.thinking[1].redacted)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
