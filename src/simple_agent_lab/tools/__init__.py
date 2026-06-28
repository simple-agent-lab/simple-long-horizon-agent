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
from pathlib import Path
from typing import Any, Callable, Literal

from simple_agent_lab.messages import ImageBlock, TextBlock


__all__ = [
    "AgentTool",
    "AbortFlag",
    "IMAGE_MIME_BY_SUFFIX",
    "Tool",
    "ToolExecuteFn",
    "ToolExecutionMode",
    "ToolResult",
    "ToolResultContent",
    "ToolUpdateFn",
    "coerce_int",
    "image_mime_for_suffix",
    "task_tool",
    "text_result",
    "tool_result_text",
    "make_read_tool",
    "make_edit_tool",
    "make_recall_tool",
]


ToolExecutionMode = Literal["sequential", "parallel"]
AbortFlag = Callable[[], bool]

ToolResultContent = tuple[TextBlock | ImageBlock, ...]

# The image extensions tools are willing to inline as attachments, mapped to the
# MIME type the LLM access layer expects. Shared by the read and bash tools so
# both honor exactly the same set; widen it here and both pick it up.
IMAGE_MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def image_mime_for_suffix(path: Path) -> str | None:
    """Return the inline-image MIME type for `path`'s suffix, or None if unsupported."""

    return IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())


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
# The canonical tool-execute signature called by `core.dispatch_tool_calls`:
#   (call_id, args, abort, on_update) -> ToolResult
# Tools that don't stream intermediate updates can ignore `on_update`;
# tools that don't support cancellation can ignore `abort`. The shape is
# fixed so static checkers can verify every `AgentTool.execute` matches.
ToolExecuteFn = Callable[
    [str, dict[str, Any], AbortFlag, ToolUpdateFn | None],
    ToolResult,
]


@dataclass(frozen=True)
class Tool:
    """Provider-facing tool definition. Pure data, safe to serialize.

    A bare `Tool` is a declaration with no local execute; use `AgentTool`
    when the runtime must actually run it.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class AgentTool(Tool):
    """A `Tool` the runtime can run: a declaration plus a required `execute`.

    `execute` uses the canonical `ToolExecuteFn` signature (above).
    `execution_mode="sequential"` runs a batch's calls one at a time;
    `"parallel"` lets the runtime dispatch them concurrently.
    `timeout_seconds` (when set) bounds a single call.
    """

    execute: ToolExecuteFn
    execution_mode: ToolExecutionMode = "parallel"
    timeout_seconds: float | None = None


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


def coerce_int(name: str, value: Any, *, minimum: int) -> int:
    """Coerce a tool argument to an ``int >= minimum``.

    Rejects `bool` (an int subclass, so `True` would otherwise read as 1) and
    fractional floats (a malformed index, not something to floor). Shared by the
    read/recall/compact tools' argument parsing; callers that accept an absent
    value guard for `None`/`""` themselves before calling.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


# Re-export after the common tool shapes are defined; `task` imports them.
from .task import task_tool  # noqa: E402
from .read import make_read_tool  # noqa: E402
from .edit import make_edit_tool  # noqa: E402
from .recall import make_recall_tool  # noqa: E402
