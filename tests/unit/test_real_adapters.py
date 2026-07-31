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

from simple_long_horizon_agent.llm import (
    LLMMessage,
    LLMRequest,
    LLMTool,
    Provider,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    complete,
    iter_stream,
)

# Test code still uses the legacy ToolCall name in places; keep an alias.
ToolCall = ToolCallBlock


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
                content=[
                    TextBlock("Calling bash."),
                    ToolCall(id="t1", name="bash", arguments={"command": "echo hello"}),
                ],
            ),
            LLMMessage(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_call_id="t1",
                        tool_name="bash",
                        content=(TextBlock("hello"),),
                    )
                ],
            ),
        ],
        tools=[_bash_tool()],
    )


@contextmanager
def _stub_module(name: str, module: types.ModuleType):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@contextmanager
def _anthropic_call(module: types.ModuleType):
    with (
        _stub_module("anthropic", module),
        mock.patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "k"}),
    ):
        yield


@contextmanager
def _openai_call(module: types.ModuleType):
    with (
        _stub_module("openai", module),
        mock.patch.dict("os.environ", {"TEST_OPENAI_KEY": "k"}),
    ):
        yield


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
    thinking_blocks: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
) -> Any:
    # Wire order matters: Anthropic's reply (and our adapter) puts thinking
    # blocks before text/tool_use, so build the stub content the same way.
    content: list[Any] = []
    for tb in thinking_blocks or []:
        block_type = "redacted_thinking" if tb.get("redacted") else "thinking"
        content.append(
            SimpleNamespace(
                type=block_type,
                thinking=tb.get("thinking", ""),
                data=tb.get("data", ""),
                signature=tb.get("signature"),
            )
        )
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
        with (
            _stub_module("anthropic", module),
            mock.patch.dict("os.environ", {}, clear=False),
        ):
            # Make sure the env var is unset.
            import os

            os.environ.pop("TEST_ANTHROPIC_KEY", None)
            with self.assertRaisesRegex(RuntimeError, "TEST_ANTHROPIC_KEY"):
                complete(req)

    def test_builds_request_and_emits_events(self) -> None:
        captured, events = _call_adapter(
            _tool_use_request(),
            _anthropic_response(
                text="Final answer.",
                tool_uses=[{"id": "t2", "name": "bash", "input": {"command": "ls"}}],
                stop_reason="tool_use",
            ),
            stream=True,
        )

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
        self.assertEqual(
            messages[0],
            {"role": "user", "content": [{"type": "text", "text": "Run: echo hello"}]},
        )
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
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(role="user", content="hi"),
                LLMMessage(role="assistant", content=""),  # empty, should drop
            ],
        )
        captured, _ = _call_adapter(req, _anthropic_response(text="hi"))
        # Only the original user message should be sent.
        self.assertEqual(len(captured["messages"]), 1)

    def test_passes_thinking_and_output_config_to_messages_create(self) -> None:
        # Reasoning has no single shape; the adapter forwards both keys
        # verbatim so the caller matches the model (adaptive thinking +
        # effort here; older models would send thinking.budget_tokens).
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
            extra={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        )
        captured, _ = _call_adapter(req, _anthropic_response())
        self.assertEqual(captured["thinking"], {"type": "adaptive"})
        self.assertEqual(captured["output_config"], {"effort": "high"})

    def _capture_reasoning(self, *, model: str, **req_kw: Any) -> dict[str, Any]:
        provider = Provider(
            id="claude-test",
            api="anthropic-messages",
            model=model,
            api_key_env="TEST_ANTHROPIC_KEY",
            **{k: v for k, v in req_kw.items() if k == "default_reasoning"},
        )
        req = LLMRequest(
            provider=provider,
            messages=[LLMMessage(role="user", content="hi")],
            **{k: v for k, v in req_kw.items() if k != "default_reasoning"},
        )
        captured, _ = _call_adapter(req, _anthropic_response())
        return captured

    def test_reasoning_configuration(self) -> None:
        cases = [
            (
                "new model uses adaptive",
                "claude-opus-4-7",
                {"reasoning": "high"},
                {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            ),
            (
                "old model uses budget",
                "claude-sonnet-4-5",
                {"reasoning": "high"},
                {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            ),
            (
                "old model clamps budget",
                "claude-sonnet-4-5",
                {"reasoning": "minimal"},
                {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            ),
            (
                "provider default applies",
                "claude-opus-4-7",
                {"default_reasoning": "medium"},
                {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "medium"},
                },
            ),
            (
                "request overrides provider default",
                "claude-opus-4-7",
                {"default_reasoning": "low", "reasoning": "high"},
                {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            ),
            (
                "raw thinking overrides normalized",
                "claude-opus-4-7",
                {
                    "reasoning": "high",
                    "extra": {"thinking": {"type": "enabled", "budget_tokens": 2048}},
                },
                {
                    "thinking": {"type": "enabled", "budget_tokens": 2048},
                    "output_config": {"effort": "high"},
                },
            ),
            ("no reasoning emits nothing", "claude-opus-4-7", {}, {}),
        ]
        for name, model, request_kwargs, expected in cases:
            with self.subTest(name):
                captured = self._capture_reasoning(model=model, **request_kwargs)
                for key in ("thinking", "output_config"):
                    if key in expected:
                        self.assertEqual(captured[key], expected[key])
                    else:
                        self.assertNotIn(key, captured)


def _stub_openai(
    response: Any, captured: dict[str, Any], *, kind: str
) -> types.ModuleType:
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


def _call_adapter(
    request: LLMRequest,
    canned_response: Any,
    *,
    stream: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Call an adapter against its fake SDK and return wire kwargs + result."""
    captured: dict[str, Any] = {}
    if request.provider.api == "anthropic-messages":
        module = _stub_anthropic(canned_response, captured)
        context = _anthropic_call(module)
    else:
        kind = "chat" if request.provider.api == "openai-chat" else "responses"
        module = _stub_openai(canned_response, captured, kind=kind)
        context = _openai_call(module)
    with context:
        result = list(iter_stream(request)) if stream else complete(request)
    return captured, result


def _chat_response(
    *,
    text: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 13,
    completion_tokens: int = 4,
    cached: int = 0,
    reasoning_content: str | None = None,
    reasoning: str | None = None,
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
    if reasoning is not None:
        message_kwargs["reasoning"] = reasoning
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
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            system_prompt="be helpful",
            messages=[
                LLMMessage(role="user", content="run ls"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ToolCall(
                            id="t1", name="bash", arguments={"command": "echo hello"}
                        )
                    ],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id="t1",
                            tool_name="bash",
                            content=(TextBlock("hello"),),
                        )
                    ],
                ),
            ],
            tools=[_bash_tool()],
        )
        captured, events = _call_adapter(
            req,
            _chat_response(
                text="Calling bash.",
                tool_calls=[
                    {
                        "id": "t2",
                        "name": "bash",
                        "arguments": json.dumps({"command": "ls"}),
                    }
                ],
                finish_reason="tool_calls",
            ),
            stream=True,
        )

        self.assertEqual(captured["__init_api_key"], "k")
        self.assertEqual(captured["model"], "gpt-test-1")

        messages = captured["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "be helpful"})
        self.assertEqual(
            messages[1],
            {"role": "user", "content": [{"type": "text", "text": "run ls"}]},
        )
        assistant_entry = messages[2]
        self.assertEqual(assistant_entry["role"], "assistant")
        self.assertIsNone(assistant_entry["content"])
        self.assertEqual(len(assistant_entry["tool_calls"]), 1)
        self.assertEqual(assistant_entry["tool_calls"][0]["id"], "t1")
        self.assertEqual(assistant_entry["tool_calls"][0]["type"], "function")
        self.assertEqual(assistant_entry["tool_calls"][0]["function"]["name"], "bash")
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
        req = LLMRequest(
            provider=OPENAI_CHAT_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        _, events = _call_adapter(
            req, _chat_response(text=None, finish_reason="length"), stream=True
        )
        kinds = [event.kind for event in events]
        # No text → no text_delta event.
        self.assertEqual(kinds, ["usage_update", "done"])
        response = events[-1].payload["response"]
        self.assertEqual(response.text, "")
        self.assertEqual(response.stop_reason, "max_tokens")

    def test_reasoning_effort_configuration(self) -> None:
        for name, request_kwargs in [
            ("raw extra", {"extra": {"reasoning_effort": "high"}}),
            ("normalized", {"reasoning": "high"}),
            (
                "raw extra overrides normalized",
                {"reasoning": "low", "extra": {"reasoning_effort": "high"}},
            ),
        ]:
            with self.subTest(name):
                request = LLMRequest(
                    provider=OPENAI_CHAT_PROVIDER,
                    messages=[LLMMessage(role="user", content="hi")],
                    **request_kwargs,
                )
                captured, _ = _call_adapter(request, _chat_response())
                self.assertEqual(captured["reasoning_effort"], "high")

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


def _responses_response(
    *,
    text_blocks: list[str] | None = None,
    function_calls: list[dict[str, Any]] | None = None,
    incomplete_reason: str | None = None,
    status: str = "completed",
    input_tokens: int = 7,
    output_tokens: int = 3,
    cached: int = 0,
    reasoning_summary: str | None = None,
    reasoning_id: str = "rs_1",
) -> Any:
    output_items: list[Any] = []
    if reasoning_summary is not None:
        output_items.append(
            SimpleNamespace(
                type="reasoning",
                id=reasoning_id,
                summary=[SimpleNamespace(type="summary_text", text=reasoning_summary)],
                content=None,
                encrypted_content=None,
                status=None,
            )
        )
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
    def _capture_replay(
        self,
        content: list[Any],
        *,
        extra: dict[str, Any] | None = None,
        provider: Provider = OPENAI_RESPONSES_PROVIDER,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        tail: str | list[ToolResultBlock] = "thanks"
        if tool_call_id:
            tail = [
                ToolResultBlock(
                    tool_call_id=tool_call_id,
                    tool_name="bash",
                    content=(TextBlock("437"),),
                )
            ]
        request = LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="multiply 23 and 19"),
                LLMMessage(role="assistant", content=content, extra=extra or {}),
                LLMMessage(role="user", content=tail),
            ],
            tools=[_bash_tool()] if tool_call_id else [],
        )
        captured, _ = _call_adapter(request, _responses_response(text_blocks=["done"]))
        return captured

    def test_reasoning_configuration(self) -> None:
        cases = [
            ("normalized", {"reasoning": "high"}),
            (
                "raw extra overrides normalized",
                {"reasoning": "low", "extra": {"reasoning": {"effort": "high"}}},
            ),
        ]
        for name, request_kwargs in cases:
            with self.subTest(name):
                request = LLMRequest(
                    provider=OPENAI_RESPONSES_PROVIDER,
                    messages=[LLMMessage(role="user", content="hi")],
                    **request_kwargs,
                )
                captured, _ = _call_adapter(
                    request, _responses_response(text_blocks=["ok"])
                )
                self.assertEqual(captured["reasoning"], {"effort": "high"})
                self.assertEqual(captured["include"], ["reasoning.encrypted_content"])

    def test_builds_request_with_instructions_and_flat_tools(self) -> None:
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            system_prompt="be brief",
            messages=[
                LLMMessage(role="user", content="hi"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ToolCall(id="prev", name="bash", arguments={"command": "ls"})
                    ],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id="prev",
                            tool_name="bash",
                            content=(TextBlock("out"),),
                        )
                    ],
                ),
            ],
            tools=[_bash_tool()],
        )
        captured, events = _call_adapter(
            req,
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
            stream=True,
        )

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
        self.assertEqual(json.loads(input_items[1]["arguments"]), {"command": "ls"})
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

    def test_inbound_reasoning_item_becomes_thinking_block(self) -> None:
        """An output ``reasoning`` item's summary text is lifted into a
        ThinkingBlock, with the item id kept on ``signature`` so the next
        outbound turn can echo the exact wire item back."""
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            messages=[LLMMessage(role="user", content="23*19?")],
        )
        _, events = _call_adapter(
            req,
            _responses_response(
                reasoning_summary="23*19 is 437.",
                reasoning_id="rs_abc",
                text_blocks=["437"],
            ),
            stream=True,
        )

        response = events[-1].payload["response"]
        self.assertEqual(len(response.thinking_blocks), 1)
        block = response.thinking_blocks[0]
        self.assertEqual(block.text, "23*19 is 437.")
        self.assertEqual(block.signature, "rs_abc")
        self.assertEqual(block.source_field, "reasoning")

    def test_outbound_replays_reasoning_item_before_tool_call(self) -> None:
        """A reasoning model served over Responses (deepseek-via-zenmux)
        rejects the next turn unless the prior reasoning item is echoed
        back ahead of the function_call it preceded."""
        captured = self._capture_replay(
            [
                ThinkingBlock(text="23*19 is 437.", signature="rs_abc"),
                ToolCall(id="c1", name="bash", arguments={"command": "echo"}),
            ],
            tool_call_id="c1",
        )

        input_items = captured["input"]
        reasoning_items = [i for i in input_items if i["type"] == "reasoning"]
        self.assertEqual(len(reasoning_items), 1)
        self.assertEqual(reasoning_items[0]["id"], "rs_abc")
        self.assertEqual(
            reasoning_items[0]["summary"],
            [{"type": "summary_text", "text": "23*19 is 437."}],
        )
        # The reasoning item must precede the function_call it belonged to.
        reasoning_pos = input_items.index(reasoning_items[0])
        call_pos = next(
            i for i, item in enumerate(input_items) if item["type"] == "function_call"
        )
        self.assertLess(reasoning_pos, call_pos)

    def test_outbound_replays_reasoning_encrypted_content(self) -> None:
        captured = self._capture_replay(
            [
                ThinkingBlock(text="23*19 is 437.", signature="rs_abc"),
                ToolCall(id="c1", name="bash", arguments={"command": "echo"}),
            ],
            extra={
                "openai_responses.reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_abc",
                        "encrypted_content": "enc_abc",
                    }
                ]
            },
            tool_call_id="c1",
        )

        reasoning_item = next(
            item for item in captured["input"] if item["type"] == "reasoning"
        )
        self.assertEqual(
            reasoning_item["summary"],
            [{"type": "summary_text", "text": "23*19 is 437."}],
        )
        self.assertEqual(reasoning_item["encrypted_content"], "enc_abc")
        self.assertEqual(captured["include"], ["reasoning.encrypted_content"])

    def test_outbound_ignores_stale_reasoning_extra_for_mismatched_signature(
        self,
    ) -> None:
        captured = self._capture_replay(
            [
                ThinkingBlock(text="23*19 is 437.", signature="rs_new"),
                TextBlock(text="437"),
            ],
            extra={
                "openai_responses.reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_old",
                        "encrypted_content": "enc_old",
                    }
                ]
            },
        )

        reasoning_item = next(
            item for item in captured["input"] if item["type"] == "reasoning"
        )
        self.assertEqual(reasoning_item["id"], "rs_new")
        self.assertEqual(
            reasoning_item["summary"],
            [{"type": "summary_text", "text": "23*19 is 437."}],
        )
        self.assertNotIn("encrypted_content", reasoning_item)

    def test_outbound_replays_encrypted_reasoning_with_empty_summary(self) -> None:
        captured = self._capture_replay(
            [ToolCall(id="c1", name="bash", arguments={"command": "echo"})],
            extra={
                "openai_responses.reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_abc",
                        "encrypted_content": "enc_abc",
                    }
                ]
            },
            tool_call_id="c1",
        )

        reasoning_item = next(
            item for item in captured["input"] if item["type"] == "reasoning"
        )
        self.assertEqual(reasoning_item["summary"], [])
        self.assertEqual(reasoning_item["encrypted_content"], "enc_abc")

    def test_outbound_replays_extra_reasoning_summary_without_encrypted_content(
        self,
    ) -> None:
        captured = self._capture_replay(
            [ToolCall(id="c1", name="bash", arguments={"command": "echo"})],
            extra={
                "openai_responses.reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_abc",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "23*19 is 437.",
                            }
                        ],
                    }
                ]
            },
            tool_call_id="c1",
        )

        reasoning_item = next(
            item for item in captured["input"] if item["type"] == "reasoning"
        )
        self.assertEqual(reasoning_item["id"], "rs_abc")
        self.assertEqual(
            reasoning_item["summary"],
            [{"type": "summary_text", "text": "23*19 is 437."}],
        )
        self.assertNotIn("encrypted_content", reasoning_item)

    def test_outbound_skips_extra_reasoning_item_without_summary_or_encrypted(
        self,
    ) -> None:
        captured = self._capture_replay(
            [ToolCall(id="c1", name="bash", arguments={"command": "echo"})],
            extra={
                "openai_responses.reasoning_items": [
                    {"type": "reasoning", "id": "rs_empty"}
                ]
            },
            tool_call_id="c1",
        )
        self.assertFalse(
            [item for item in captured["input"] if item["type"] == "reasoning"]
        )

    def test_outbound_skips_orphan_extra_reasoning_item(self) -> None:
        captured = self._capture_replay(
            [],
            extra={
                "openai_responses.reasoning_items": [
                    {
                        "type": "reasoning",
                        "id": "rs_abc",
                        "encrypted_content": "enc_abc",
                    }
                ]
            },
        )

        self.assertEqual(
            [item["type"] for item in captured["input"]],
            ["message", "message"],
        )
        self.assertFalse(
            [item for item in captured["input"] if item["type"] == "reasoning"]
        )

    def test_outbound_skips_reasoning_when_replay_disabled(self) -> None:
        """Endpoints that manage continuity server-side can disable replay."""
        provider = Provider(
            id="gpt-resp-noreplay",
            api="openai-responses",
            model="gpt-test-1",
            api_key_env="TEST_OPENAI_KEY",
            replay_reasoning=False,
        )
        captured = self._capture_replay(
            [
                ThinkingBlock(text="23*19 is 437.", signature="rs_abc"),
                TextBlock(text="437"),
            ],
            provider=provider,
        )

        self.assertFalse([i for i in captured["input"] if i["type"] == "reasoning"])
        self.assertNotIn("include", captured)

    def test_explicit_include_is_preserved_when_replay_disabled(self) -> None:
        provider = Provider(
            id="gpt-resp-noreplay",
            api="openai-responses",
            model="gpt-test-1",
            api_key_env="TEST_OPENAI_KEY",
            replay_reasoning=False,
        )
        req = LLMRequest(
            provider=provider,
            messages=[LLMMessage(role="user", content="hi")],
            extra={"include": ["message.input_image.image_url"]},
        )
        captured, _ = _call_adapter(req, _responses_response(text_blocks=["done"]))

        self.assertEqual(captured["include"], ["message.input_image.image_url"])

    def test_incomplete_max_tokens_maps_to_max_tokens_stop(self) -> None:
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        _, response = _call_adapter(
            req,
            _responses_response(
                text_blocks=["partial"],
                incomplete_reason="max_output_tokens",
                status="incomplete",
            ),
        )
        self.assertEqual(response.stop_reason, "max_tokens")

    def test_passes_extra_headers_to_responses_create(self) -> None:
        req = LLMRequest(
            provider=OPENAI_RESPONSES_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
            extra={
                "extra_headers": {
                    "extra": '{"session_id":"s"}',
                    "X-TT-logid": "l",
                }
            },
        )
        captured, _ = _call_adapter(req, _responses_response(text_blocks=["ok"]))

        self.assertEqual(
            captured["extra_headers"],
            {"extra": '{"session_id":"s"}', "X-TT-logid": "l"},
        )


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
    def _assistant_wire(
        self, *, replay: bool = True, source_field: str | None = None
    ) -> dict[str, Any]:
        provider = OPENAI_CHAT_PROVIDER
        if not replay:
            provider = Provider(
                id="gpt-chat-test",
                api="openai-chat",
                model="gpt-test-1",
                api_key_env="TEST_OPENAI_KEY",
                replay_reasoning=False,
            )
        request = LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="Compute 2*3."),
                LLMMessage(
                    role="assistant",
                    content=[
                        ThinkingBlock(text="2*3 is 6.", source_field=source_field),
                        TextBlock(text="6"),
                    ],
                ),
                LLMMessage(role="user", content="now double it"),
            ],
        )
        captured, _ = _call_adapter(request, _chat_response())
        return next(m for m in captured["messages"] if m["role"] == "assistant")

    def test_inbound_reasoning_fields_become_thinking_blocks(self) -> None:
        for field in ("reasoning_content", "reasoning"):
            with self.subTest(field):
                request = LLMRequest(
                    provider=OPENAI_CHAT_PROVIDER,
                    messages=[LLMMessage(role="user", content="2*3?")],
                )
                _, events = _call_adapter(
                    request,
                    _chat_response(
                        text="Result: 6.",
                        **{field: "2*3 is 6."},
                    ),
                    stream=True,
                )
                kinds = [event.kind for event in events]
                self.assertLess(
                    kinds.index("thinking_delta"), kinds.index("text_delta")
                )
                response = events[-1].payload["response"]
                self.assertEqual(response.thinking_blocks[0].text, "2*3 is 6.")
                self.assertEqual(response.text, "Result: 6.")
                self.assertEqual(
                    [block.kind for block in response.content],
                    ["thinking", "text"],
                )
                if field == "reasoning":
                    self.assertEqual(
                        response.thinking_blocks[0].source_field, "reasoning"
                    )

    def test_outbound_uses_canonical_reasoning_content(self) -> None:
        for source_field in (None, "reasoning"):
            with self.subTest(source_field):
                assistant = self._assistant_wire(source_field=source_field)
                self.assertEqual(assistant["reasoning_content"], "2*3 is 6.")
                self.assertEqual(assistant["content"], "6")
                self.assertNotIn("reasoning", assistant)

    def test_outbound_skips_replay_when_opted_out(self) -> None:
        self.assertNotIn("reasoning_content", self._assistant_wire(replay=False))


class AnthropicReasoningReplayTest(unittest.TestCase):
    _RESPONSE_KW = {
        "text": "done",
        "thinking_blocks": [{"thinking": "careful step", "signature": "sig-1"}],
    }

    def test_inbound_captures_thinking_with_signature(self) -> None:
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[LLMMessage(role="user", content="hi")],
        )
        _, response = _call_adapter(req, _anthropic_response(**self._RESPONSE_KW))

        self.assertEqual(len(response.thinking_blocks), 1)
        block = response.thinking_blocks[0]
        self.assertEqual(block.text, "careful step")
        self.assertEqual(block.signature, "sig-1")
        self.assertFalse(block.redacted)
        self.assertEqual([b.kind for b in response.content], ["thinking", "text"])

    def test_outbound_replays_thinking_block_first(self) -> None:
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(role="user", content="go"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ThinkingBlock(text="careful step", signature="sig-1"),
                        TextBlock(text="done"),
                        ToolCall(id="t1", name="bash", arguments={"command": "ls"}),
                    ],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id="t1",
                            tool_name="bash",
                            content=(TextBlock("out"),),
                        )
                    ],
                ),
            ],
        )
        captured, _ = _call_adapter(req, _anthropic_response(**self._RESPONSE_KW))

        assistant_msg = next(
            m for m in captured["messages"] if m["role"] == "assistant"
        )
        blocks = assistant_msg["content"]
        self.assertEqual(blocks[0]["type"], "thinking")
        self.assertEqual(blocks[0]["thinking"], "careful step")
        self.assertEqual(blocks[0]["signature"], "sig-1")
        self.assertEqual(blocks[1]["type"], "text")
        self.assertEqual(blocks[2]["type"], "tool_use")

    def test_outbound_skips_replay_when_opted_out(self) -> None:
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
                    content=[ThinkingBlock(text="x", signature="s")],
                ),
                LLMMessage(role="user", content="again"),
            ],
        )
        captured, _ = _call_adapter(req, _anthropic_response(**self._RESPONSE_KW))

        for msg in (m for m in captured["messages"] if m["role"] == "assistant"):
            for block in msg["content"]:
                self.assertNotEqual(block["type"], "thinking")


class BridgeThinkingPreservationTest(unittest.TestCase):
    def test_thinking_blocks_carry_signature_into_assistant_message(self) -> None:
        from simple_long_horizon_agent.llm.bridge import (
            llm_response_to_assistant_message,
        )
        from simple_long_horizon_agent.llm.types import LLMResponse
        from simple_long_horizon_agent.messages import AssistantMessage

        response = LLMResponse(
            content=[
                ThinkingBlock(text="step 1", signature="s1"),
                ThinkingBlock(text="step 2", signature="s2", redacted=True),
                TextBlock(text="answer"),
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

    def test_responses_reasoning_metadata_rides_in_message_extra(self) -> None:
        from simple_long_horizon_agent.llm.bridge import (
            llm_response_to_assistant_message,
            message_to_llm_message,
        )
        from simple_long_horizon_agent.llm.types import LLMResponse
        from simple_long_horizon_agent.messages import AssistantMessage

        summary = [SimpleNamespace(type="summary_text", text="step 1")]
        expected_summary = [{"type": "summary_text", "text": "step 1"}]
        cases = [
            ("encrypted only", {"encrypted_content": "enc_1"}),
            (
                "summary and encrypted",
                {"summary": summary, "encrypted_content": "enc_1"},
            ),
            ("summary only", {"summary": summary, "encrypted_content": None}),
        ]
        for name, metadata in cases:
            with self.subTest(name):
                raw_item = SimpleNamespace(type="reasoning", id="rs_1", **metadata)
                response = LLMResponse(
                    content=[ThinkingBlock(text="step 1", signature="rs_1")],
                    raw={"response": SimpleNamespace(output=[raw_item])},
                )
                msg = llm_response_to_assistant_message(
                    response, sender="agent", target="user", kind="final"
                )
                assert isinstance(msg, AssistantMessage)
                expected = {"type": "reasoning", "id": "rs_1"}
                if "summary" in metadata:
                    expected["summary"] = expected_summary
                if metadata.get("encrypted_content"):
                    expected["encrypted_content"] = "enc_1"
                projected = message_to_llm_message(msg)
                self.assertEqual(
                    projected.extra["openai_responses.reasoning_items"],
                    [expected],
                )


class MessageExtraTest(unittest.TestCase):
    """`LLMMessage.extra` is the per-message twin of `LLMRequest.extra`.

    Adapters read provider-namespaced keys; unknown namespaces are
    silently ignored so a transcript stays portable across providers.
    """

    def test_anthropic_cache_breakpoint_attaches_cache_control(self) -> None:
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(
                    role="user",
                    content="cache this",
                    extra={"anthropic.cache_breakpoint": True},
                ),
            ],
        )
        captured, _ = _call_adapter(req, _anthropic_response())

        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        last_block = user_msg["content"][-1]
        self.assertEqual(last_block.get("cache_control"), {"type": "ephemeral"})

    def test_unknown_namespace_silently_ignored(self) -> None:
        req = LLMRequest(
            provider=ANTHROPIC_PROVIDER,
            messages=[
                LLMMessage(
                    role="user",
                    content="hi",
                    extra={"openai.name": "alice", "gemini.safety": "low"},
                ),
            ],
        )
        captured, _ = _call_adapter(req, _anthropic_response())

        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        for block in user_msg["content"]:
            self.assertNotIn("cache_control", block)

    def test_bridge_lifts_runtime_data_extra_to_llm_message(self) -> None:
        from simple_long_horizon_agent.llm.bridge import message_to_llm_message
        from simple_long_horizon_agent.messages import user_message

        msg = user_message(
            "hello",
            sidecar={
                "extra": {"anthropic.cache_breakpoint": True, "openai.name": "bob"}
            },
        )
        llm = message_to_llm_message(msg)
        self.assertEqual(
            dict(llm.extra),
            {"anthropic.cache_breakpoint": True, "openai.name": "bob"},
        )


class ParallelToolResultBundleTest(unittest.TestCase):
    """A user message bundling N ToolResultBlocks renders correctly per provider.

    Anthropic wants one user message with N tool_result content blocks (so
    the model sees them as a parallel batch). OpenAI-Chat wants N separate
    `role="tool"` entries (each with its own tool_call_id).
    """

    def _bundled_request(self, provider: Provider) -> LLMRequest:
        return LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="run two things"),
                LLMMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock("a", "bash", {"command": "echo aaa"}),
                        ToolCallBlock("b", "bash", {"command": "echo bbb"}),
                    ],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id="a",
                            tool_name="bash",
                            content=(TextBlock("out-a"),),
                        ),
                        ToolResultBlock(
                            tool_call_id="b",
                            tool_name="bash",
                            content=(TextBlock("out-b"),),
                        ),
                    ],
                ),
            ],
        )

    def test_anthropic_bundles_into_one_user_wire_message(self) -> None:
        captured, _ = _call_adapter(
            self._bundled_request(ANTHROPIC_PROVIDER), _anthropic_response(text="done")
        )

        user_msgs = [m for m in captured["messages"] if m["role"] == "user"]
        bundles = [
            m
            for m in user_msgs
            if any(b.get("type") == "tool_result" for b in m["content"])
        ]
        self.assertEqual(len(bundles), 1)
        tool_blocks = [b for b in bundles[0]["content"] if b["type"] == "tool_result"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(
            {b["tool_use_id"] for b in tool_blocks},
            {"a", "b"},
        )

    def test_openai_chat_splits_into_n_tool_wire_entries(self) -> None:
        captured, _ = _call_adapter(
            self._bundled_request(OPENAI_CHAT_PROVIDER), _chat_response(text="done")
        )

        tool_entries = [m for m in captured["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_entries), 2)
        self.assertEqual(
            {m["tool_call_id"] for m in tool_entries},
            {"a", "b"},
        )
        contents = {m["tool_call_id"]: m["content"] for m in tool_entries}
        self.assertEqual(contents["a"], "out-a")
        self.assertEqual(contents["b"], "out-b")


class MultimodalToolResultTest(unittest.TestCase):
    """A tool result carrying an image renders correctly per provider.

    Anthropic accepts a list of `text` / `image` blocks directly inside
    `tool_result.content`. OpenAI Chat / Responses don't (the role=tool
    content is a string), so the adapter surfaces the image in an
    adjacent `role="user"` entry tagged with the tool name.
    """

    IMAGE_DATA = "ZmFrZS1wbmctYnl0ZXM="  # base64 of "fake-png-bytes"

    def _request_with_image(self, provider: Provider) -> LLMRequest:
        from simple_long_horizon_agent.messages import ImageBlock

        return LLMRequest(
            provider=provider,
            messages=[
                LLMMessage(role="user", content="screenshot please"),
                LLMMessage(
                    role="assistant",
                    content=[ToolCallBlock("c1", "screenshot", {})],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id="c1",
                            tool_name="screenshot",
                            content=(
                                TextBlock("captured"),
                                ImageBlock(data=self.IMAGE_DATA, mime_type="image/png"),
                            ),
                        ),
                    ],
                ),
            ],
        )

    def test_anthropic_carries_image_inside_tool_result_content(self) -> None:
        captured, _ = _call_adapter(
            self._request_with_image(ANTHROPIC_PROVIDER), _anthropic_response()
        )

        bundle = next(
            m
            for m in captured["messages"]
            if m["role"] == "user"
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        tool_block = next(b for b in bundle["content"] if b["type"] == "tool_result")
        # Anthropic accepts the list form — text + image blocks side by side.
        self.assertIsInstance(tool_block["content"], list)
        kinds = [b["type"] for b in tool_block["content"]]
        self.assertEqual(kinds, ["text", "image"])
        image = tool_block["content"][1]
        self.assertEqual(image["source"]["data"], self.IMAGE_DATA)
        self.assertEqual(image["source"]["media_type"], "image/png")

    def test_openai_chat_splits_image_into_adjacent_user_message(self) -> None:
        captured, _ = _call_adapter(
            self._request_with_image(OPENAI_CHAT_PROVIDER),
            _chat_response(text="red"),
        )

        wire = captured["messages"]
        # Find the role=tool entry — content is a plain string with no image.
        tool_entries = [m for m in wire if m["role"] == "tool"]
        self.assertEqual(len(tool_entries), 1)
        self.assertIsInstance(tool_entries[0]["content"], str)
        self.assertIn("captured", tool_entries[0]["content"])
        # The adjacent user entry should follow with the image inlined.
        tool_idx = wire.index(tool_entries[0])
        follow = wire[tool_idx + 1]
        self.assertEqual(follow["role"], "user")
        types = [part["type"] for part in follow["content"]]
        self.assertIn("text", types)
        self.assertIn("image_url", types)
        image_part = next(p for p in follow["content"] if p["type"] == "image_url")
        self.assertIn(self.IMAGE_DATA, image_part["image_url"]["url"])

    def test_openai_responses_splits_image_into_adjacent_user_item(self) -> None:
        captured, _ = _call_adapter(
            self._request_with_image(OPENAI_RESPONSES_PROVIDER),
            _responses_response(text_blocks=["red"]),
        )

        items = captured["input"]
        # function_call_output stays text-only.
        outputs = [it for it in items if it.get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertIn("captured", outputs[0]["output"])
        # And there is a user message item carrying the image.
        out_idx = items.index(outputs[0])
        follow = items[out_idx + 1]
        self.assertEqual(follow["type"], "message")
        self.assertEqual(follow["role"], "user")
        types = [part["type"] for part in follow["content"]]
        self.assertIn("input_image", types)
        image_part = next(p for p in follow["content"] if p["type"] == "input_image")
        self.assertIn(self.IMAGE_DATA, image_part["image_url"])


class RawCaptureTest(unittest.TestCase):
    """`LLMResponse.raw` records the SDK call's request + response.

    The request snapshot preserves the full outbound body including the
    messages / input history so it can serve as a faithful HTTP-level
    replay record.
    """

    def test_raw_preserves_outbound_history(self) -> None:
        cases = [
            (
                "openai chat",
                OPENAI_CHAT_PROVIDER,
                _chat_response(),
                "messages",
                "be brief",
                2,
            ),
            (
                "anthropic",
                ANTHROPIC_PROVIDER,
                _anthropic_response(),
                "messages",
                None,
                1,
            ),
            (
                "openai responses",
                OPENAI_RESPONSES_PROVIDER,
                _responses_response(text_blocks=["ok"]),
                "input",
                None,
                1,
            ),
        ]
        for name, provider, canned, history_key, system_prompt, expected_len in cases:
            with self.subTest(name):
                request = LLMRequest(
                    provider=provider,
                    system_prompt=system_prompt,
                    messages=[LLMMessage(role="user", content="hello")],
                )
                _, response = _call_adapter(request, canned)
                snapshot = response.raw["request"]
                self.assertIsInstance(snapshot[history_key], list)
                self.assertEqual(len(snapshot[history_key]), expected_len)
                self.assertTrue(response.raw["response"])
                if provider is OPENAI_CHAT_PROVIDER:
                    self.assertEqual(snapshot["model"], "gpt-test-1")
                    self.assertIn("temperature", snapshot)


class CachedTokenNormalizationTest(unittest.TestCase):
    """Cache tokens must be additive to `input_tokens`, never a subset.

    OpenAI reports `prompt_tokens` *including* the cached portion, with
    `cached_tokens` as a sub-breakdown. If the adapter stored both verbatim,
    `context_tokens` (= input + output + cache_read + cache_write) would count
    the cache twice — inflating the context-window estimate that drives
    compression. The numbers below are from a real warm-cache call: a 2007-token
    prompt served almost entirely from cache.
    """

    def test_cache_tokens_are_normalized_as_additive(self) -> None:
        cases = [
            (
                "openai chat subtracts cache",
                OPENAI_CHAT_PROVIDER,
                _chat_response(prompt_tokens=2007, completion_tokens=90, cached=1984),
            ),
            (
                "openai responses subtracts cache",
                OPENAI_RESPONSES_PROVIDER,
                _responses_response(
                    text_blocks=["ok"],
                    input_tokens=2007,
                    output_tokens=90,
                    cached=1984,
                ),
            ),
            (
                "anthropic already reports additive input",
                ANTHROPIC_PROVIDER,
                _anthropic_response(input_tokens=23, output_tokens=90, cache_read=1984),
            ),
        ]
        for name, provider, canned in cases:
            with self.subTest(name):
                request = LLMRequest(
                    provider=provider,
                    messages=[LLMMessage(role="user", content="hi")],
                )
                _, response = _call_adapter(request, canned)
                self.assertEqual(response.usage.input_tokens, 23)
                self.assertEqual(response.usage.cache_read_tokens, 1984)
                self.assertEqual(response.usage.context_tokens, 2097)
