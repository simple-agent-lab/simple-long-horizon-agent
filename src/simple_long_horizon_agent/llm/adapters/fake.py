"""Deterministic fake adapter.

Useful for tests, examples, and exercising the streaming protocol
without a network call. By default it derives a small, predictable response
from `LLMRequest.system_prompt`, `LLMRequest.messages`, and `LLMRequest.tools`.
That keeps examples honest: callers pass the same messages they would pass to a
real model.

`LLMRequest.extra` only controls streaming mechanics:

    extra["chunk_size"] : int    — chars per text_delta (default: 5)
    extra["delay"]      : float  — sleep before each event, simulating
                                   network pacing (default: 0)
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterator

from ...messages import ContentBlock, TextBlock, ToolCallBlock
from ..stream import register_adapter
from ..types import LLMRequest, LLMResponse, StreamEvent, TokenUsage


_DEFAULT_TEXT = "Fake response generated from the provided messages."


def stream(req: LLMRequest) -> Iterator[StreamEvent]:
    extra = req.extra
    text, tool_calls = _complete_from_request(req)
    chunk: int = max(1, int(extra.get("chunk_size", 5)))
    delay: float = float(extra.get("delay", 0.0))

    def _pace() -> None:
        if delay > 0:
            time.sleep(delay)

    for i in range(0, len(text), chunk):
        _pace()
        yield StreamEvent(kind="text_delta", payload={"delta": text[i : i + chunk]})

    # Tool-call phase. Emit start + complete; no JSON-arg streaming for the fake.
    for tc in tool_calls:
        _pace()
        yield StreamEvent(kind="tool_call_start", payload={"tool_call": tc})
        _pace()
        yield StreamEvent(kind="tool_call_complete", payload={"tool_call": tc})

    # Usage report. Token counts are rough estimates; this is a fake.
    usage = TokenUsage(
        input_tokens=sum(_estimate_tokens(m) for m in req.messages),
        output_tokens=max(1, len(text) // 4),
    )
    yield StreamEvent(kind="usage_update", payload={"usage": usage})

    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_calls)
    response = LLMResponse(
        content=tuple(blocks),
        stop_reason="tool_use" if tool_calls else "end_turn",
        # Echo the requested model like a real provider would, so the fake
        # exercises the same served-model path instead of relying on
        # complete()'s fallback.
        model=req.provider.model,
        usage=usage,
    )
    yield StreamEvent(kind="done", payload={"response": response})


def _complete_from_request(req: LLMRequest) -> tuple[str, list[ToolCallBlock]]:
    text = _request_text(req)
    lower = text.lower()
    tool_names = {tool.name for tool in req.tools}

    if "bash" in tool_names:
        if _has_tool_result(req):
            return _bash_answer(req), []
        return "I'll run the requested command with bash.", [
            ToolCallBlock(
                "bash_1",
                "bash",
                {
                    "command": _bash_command_from_request(req),
                    "description": "Run the requested local shell command.",
                },
            )
        ]

    if "run_agent" in tool_names:
        if _has_tool_result(req):
            return _last_tool_result_text(req), []
        return "I'll ask a focused subagent to handle that.", [
            ToolCallBlock(
                "agent_1",
                "run_agent",
                {
                    "agent_name": "tweet_writer",
                    "prompt": _agent_tool_prompt(req),
                },
            )
        ]

    if "calculate" in tool_names:
        if _has_tool_result(req):
            return _shopping_answer(req), []
        expression = _shopping_expression(text)
        if expression:
            return "Let me work that out.", [
                ToolCallBlock("call_1", "calculate", {"expression": expression})
            ]

    if "get_weather" in tool_names:
        city = _city_from_text(text) or "Tokyo"
        if _has_tool_result(req):
            forecast = _last_tool_result_text(req) or "weather data unavailable"
            return (
                f"It's {forecast} in {city} today — a light jacket should be enough.",
                [],
            )
        return "Let me check the weather first.", [
            ToolCallBlock("wx_1", "get_weather", {"city": city})
        ]

    prompt = (req.system_prompt or "").lower()
    if "break" in prompt and "bullet" in prompt and "decorator" in lower:
        return (
            "- decorators wrap functions\n"
            "- @-syntax keeps call sites clean\n"
            "- common uses: caching, auth, timing"
        ), []
    if "turn an outline" in prompt and "decorator" in lower:
        return (
            "Decorators in Python wrap functions to add behavior — caching, auth, "
            "timing — using a tidy @-syntax. They are a clean way to factor "
            "cross-cutting concerns out of business logic."
        ), []
    if "polish" in prompt and "tweet" in prompt and "decorator" in lower:
        if "casual" in lower:
            return (
                "py decorators 🪄 1-liner magic: caching, auth, timing — slap @ on top, profit."
            ), []
        return (
            "Python decorators wrap functions in one line to add caching, auth, "
            "or timing — a clean way to factor cross-cutting concerns."
        ), []

    last_user = _last_message_text(req, role="user")
    if last_user:
        return f"Fake response to: {last_user}", []
    return _DEFAULT_TEXT, []


def _request_text(req: LLMRequest) -> str:
    parts = [req.system_prompt or ""]
    parts.extend(_message_text(message) for message in req.messages)
    return "\n".join(part for part in parts if part)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return _strip_routing_header(content)
    if isinstance(content, (list, tuple)):
        return " ".join(
            _strip_routing_header(getattr(block, "text", "")) for block in content
        )
    return str(content)


def _strip_routing_header(text: str) -> str:
    if text.startswith("[") and "]\n" in text:
        return text.split("]\n", 1)[1]
    return text


def _last_message_text(req: LLMRequest, *, role: str | None = None) -> str:
    for message in reversed(req.messages):
        if role is not None and getattr(message, "role", "") != role:
            continue
        text = _message_text(message)
        if text:
            return text
    return ""


def _has_tool_result(req: LLMRequest) -> bool:
    return any(getattr(message, "tool_results", ()) for message in req.messages)


def _last_tool_result_text(req: LLMRequest) -> str:
    from ...messages import text_of

    for message in reversed(req.messages):
        results = getattr(message, "tool_results", ())
        if results:
            return text_of(results[-1].content)
    return ""


def _agent_tool_prompt(req: LLMRequest) -> str:
    task = _last_message_text(req, role="user") or _request_text(req)
    if "casual" not in task.lower():
        task = f"{task} Make it casual."
    return task


def _bash_command_from_request(req: LLMRequest) -> str:
    text = _last_message_text(req, role="user") or _request_text(req)
    patterns = (
        r"command:\s*`([^`]+)`",
        r"<bash>(.*?)</bash>",
        r"^\s*\$\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            command = match.group(1).strip()
            if command:
                return command
    return "pwd && find src/simple_long_horizon_agent -maxdepth 1 -type f -name '*.py' | sort"


def _bash_answer(req: LLMRequest) -> str:
    result = _last_tool_result_text(req)
    return f"Bash observation:\n{result}" if result else "Bash completed."


def _shopping_expression(text: str) -> str:
    pattern = re.compile(
        r"(?P<shirts>\d+)\s+shirts?\s+at\s+\$(?P<shirt_price>\d+(?:\.\d+)?)"
        r".*?(?P<pants>\d+)\s+pairs?\s+of\s+pants\s+at\s+\$(?P<pant_price>\d+(?:\.\d+)?)"
        r".*?(?P<tax>\d+(?:\.\d+)?)%\s+tax",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    multiplier = 1 + float(match.group("tax")) / 100
    return (
        f"({match.group('shirts')} * {match.group('shirt_price')} + "
        f"{match.group('pants')} * {match.group('pant_price')}) * "
        f"{multiplier:.3f}".rstrip("0").rstrip(".")
    )


def _shopping_answer(req: LLMRequest) -> str:
    result = _last_tool_result_text(req)
    match = re.search(r"=\s*(-?\d+(?:\.\d+)?)", result)
    if match:
        total = round(float(match.group(1)) + 1e-9, 2)
        return (
            f"Your total is ${total:.2f} "
            "(3 shirts at $19.99 + 2 pants at $34.50, plus 8.5% tax)."
        )
    return "Your total is ready; check the tool result for the exact amount."


def _city_from_text(text: str) -> str:
    for city in ("Tokyo", "New York", "London"):
        if city.lower() in text.lower():
            return city
    return ""


def _estimate_tokens(message: Any) -> int:
    return max(1, len(_message_text(message)) // 4)


register_adapter("fake", stream)
