"""Shared tool data shapes for Simple Agent Lab.

Agent loops own dispatch semantics. This module owns the common values that
cross the tool boundary: tool definitions, tool results, and small helpers for
turning result blocks into model-visible text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


__all__ = [
    "AgentTool",
    "AbortFlag",
    "Tool",
    "ToolContent",
    "ToolExecuteFn",
    "ToolExecutionMode",
    "ToolResult",
    "ToolUpdateFn",
    "text_result",
    "tool_result_text",
]


ToolExecutionMode = Literal["sequential", "parallel"]
AbortFlag = Callable[[], bool]


@dataclass(frozen=True)
class ToolContent:
    """One model-visible content block returned by a tool."""

    kind: Literal["text", "image"] = "text"
    text: str = ""
    image_url: str = ""


@dataclass(frozen=True)
class ToolResult:
    """A tool's return value.

    `content` is what the model sees on the next turn. `details` is for local
    inspection or UI and never crosses the model boundary. `is_error=True`
    means the error text should go back to the model so it can self-correct.
    `terminate=True` asks the owning runtime to stop after recording the result.
    """

    content: list[ToolContent] = field(default_factory=list)
    details: Any = None
    is_error: bool = False
    terminate: bool = False


ToolUpdateFn = Callable[[ToolResult], None]
ToolExecuteFn = Callable[..., ToolResult]


@dataclass(frozen=True)
class Tool:
    """Provider-facing tool definition. Pure data, safe to serialize."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class AgentTool(Tool):
    """A tool definition plus local execution metadata.

    Runtimes may call `execute` with different signatures. For example, the
    functional loop calls `(args)`, the event runtime calls `(call_id, args)`,
    and the balanced runtime calls `(call_id, args, abort, on_update)`.
    """

    execute: Optional[ToolExecuteFn] = None
    label: str = ""
    execution_mode: ToolExecutionMode = "parallel"
    timeout_seconds: Optional[float] = None


def text_result(
    text: str,
    *,
    details: Any = None,
    is_error: bool = False,
    terminate: bool = False,
) -> ToolResult:
    """Build a text-only `ToolResult`."""

    return ToolResult(
        content=[ToolContent(text=text)],
        details=details,
        is_error=is_error,
        terminate=terminate,
    )


def tool_result_text(result: ToolResult) -> str:
    """Return the model-visible text blocks from a tool result."""

    return "\n".join(block.text for block in result.content if block.kind == "text")
