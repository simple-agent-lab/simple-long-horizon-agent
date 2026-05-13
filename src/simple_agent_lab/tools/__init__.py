"""Shared tool data shapes for Simple Agent Lab.

Agent loops own dispatch semantics. This package owns the common values that
cross the tool boundary: tool definitions, tool results, and small helpers for
turning result blocks into model-visible text.

`ToolResult.content` is the same runtime block tuple the rest of the codebase
uses (`TextBlock | ImageBlock`), so tool authors return blocks that flow
through to the LLM access layer without an intermediate translation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from simple_agent_lab.messages import ImageBlock, TextBlock


__all__ = [
    "AgentTool",
    "AbortFlag",
    "Tool",
    "ToolExecuteFn",
    "ToolExecutionMode",
    "ToolResult",
    "ToolResultContent",
    "ToolUpdateFn",
    "text_result",
    "tool_result_text",
]


ToolExecutionMode = Literal["sequential", "parallel"]
AbortFlag = Callable[[], bool]

ToolResultContent = tuple[TextBlock | ImageBlock, ...]


@dataclass(frozen=True)
class ToolResult:
    """A tool's return value.

    `content` is what the model sees on the next turn — a tuple of visible
    blocks (text or image) ready to flow into a `ToolResultBlock` without
    intermediate translation. `details` is for local inspection or UI and
    never crosses the model boundary. `is_error=True` means the error text
    should go back to the model so it can self-correct. `terminate=True`
    asks the owning runtime to stop after recording the result.
    """

    content: ToolResultContent = ()
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

    content: ToolResultContent = (TextBlock(text),) if text else ()
    return ToolResult(
        content=content,
        details=details,
        is_error=is_error,
        terminate=terminate,
    )


def tool_result_text(result: ToolResult) -> str:
    """Concatenate the text blocks in a tool result."""

    return "\n".join(
        block.text for block in result.content if isinstance(block, TextBlock)
    )
