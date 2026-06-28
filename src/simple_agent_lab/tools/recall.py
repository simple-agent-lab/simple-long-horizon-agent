"""Recall tool: retrieve compressed-away transcript messages by index.

Compression folds older messages out of the active view, but `State` is
append-only — the originals never leave `state.messages`. When a compression
replacement cites the indices it folded ("[Compressed from transcript messages
2-8 ...]"), this tool is the matching retrieval side: the model reads the
citation in a summary and fetches the originals it needs verbatim. Together they
turn
compression from lossy deletion into recoverable externalization — an agent
that can recall what a summary dropped can afford aggressive compaction.

The factory closes over the `State` it serves, so build the tool next to the
state and drive the loop with the module-level `run`:

    state = State(task)
    state.send("task", "user", "worker", task)
    agent = Agent("worker", step, tools=(make_recall_tool(state),))
    for _ in run(agent, state):
        ...

The tool is read-only — it renders `state.messages` entries and records
nothing — so it is safe in the parallel tool pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simple_agent_lab.messages import (
    ImageBlock,
    Message,
    message_tool_calls,
    text_of,
    tool_results_of,
)

from . import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, coerce_int, text_result

if TYPE_CHECKING:
    from simple_agent_lab.state import State

RECALL_TOOL_NAME = "recall"

# A recalled message goes straight back into the context the compression just
# shrank, so the output is capped on two axes: per message (each rendered
# message) and per call (the whole batch). The per-call cap is the important
# one — without it `max_chars_per_message * max_indices` worth of text could be
# re-injected in a single recall, undoing the compaction that just ran.
DEFAULT_MAX_CHARS_PER_MESSAGE = 4000
DEFAULT_MAX_TOTAL_CHARS = 8000
DEFAULT_MAX_INDICES = 20


def make_recall_tool(
    state: "State",
    *,
    max_chars_per_message: int = DEFAULT_MAX_CHARS_PER_MESSAGE,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    max_indices: int = DEFAULT_MAX_INDICES,
    tool_name: str = RECALL_TOOL_NAME,
) -> AgentTool:
    """Return an `AgentTool` that reads original transcript messages off `state`."""
    if max_chars_per_message <= 0:
        raise ValueError("max_chars_per_message must be > 0")
    if max_total_chars < max_chars_per_message:
        raise ValueError("max_total_chars must be >= max_chars_per_message")
    if max_indices <= 0:
        raise ValueError("max_indices must be > 0")

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        try:
            indices = _coerce_indices(args.get("indices"), max_indices=max_indices)
        except ValueError as exc:
            return text_result(f"Invalid recall argument: {exc}", is_error=True)
        messages = state.messages
        invalid = [index for index in indices if not 0 <= index < len(messages)]
        if invalid:
            return text_result(
                f"Index(es) {invalid} out of range; the transcript currently "
                f"has {len(messages)} messages (valid indices "
                f"0-{len(messages) - 1}).",
                is_error=True,
            )
        rendered: list[str] = []
        used = 0
        returned = indices
        for position, index in enumerate(indices):
            block = render_transcript_message(
                index, messages[index], max_chars=max_chars_per_message
            )
            cost = len(block) + (2 if rendered else 0)  # 2 for the "\n\n" joiner
            # Always return the first message (already per-message capped); stop
            # before the batch as a whole would exceed the per-call budget.
            if rendered and used + cost > max_total_chars:
                remaining = len(indices) - position
                rendered.append(
                    f"… [recall truncated: {remaining} more message(s) not "
                    f"shown; this call's {max_total_chars}-char budget is full. "
                    "Request the rest in a follow-up recall.]"
                )
                returned = indices[:position]
                break
            rendered.append(block)
            used += cost
        return text_result("\n\n".join(rendered), details={"indices": returned})

    return AgentTool(
        name=tool_name,
        description=(
            "Retrieve original transcript messages that were compressed out of "
            "your context. Compression summaries cite their sources, e.g. "
            "'[Compressed from transcript messages 2-8 ...]' — pass those "
            "indices here to read the originals verbatim. Use it when a "
            "summary is missing a detail you now need."
        ),
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Transcript message indices to retrieve (0-based, as "
                        "cited by a compression summary)."
                    ),
                },
            },
            "required": ["indices"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def render_transcript_message(index: int, message: Message, *, max_chars: int) -> str:
    """Render one transcript message for the model: header plus visible content."""
    header = (
        f"[transcript message {index}] role={message.role} "
        f"sender={message.sender} target={message.target} kind={message.kind}"
    )
    parts: list[str] = []
    text = text_of(message.content)
    if text:
        parts.append(text)
    for call in message_tool_calls(message):
        parts.append(f"tool_call {call.id} -> {call.name}({dict(call.arguments)!r})")
    for block in tool_results_of(message.content):
        label = "tool_result(error)" if block.is_error else "tool_result"
        parts.append(
            f"{label} {block.tool_call_id} ({block.tool_name}): "
            f"{text_of(block.content)}"
        )
    images = sum(1 for block in message.content if isinstance(block, ImageBlock))
    images += sum(
        1
        for result in tool_results_of(message.content)
        for block in result.content
        if isinstance(block, ImageBlock)
    )
    if images:
        parts.append(f"({images} image block(s) omitted)")
    body = "\n".join(parts) or "(no visible content)"
    total = len(body)
    if total > max_chars:
        body = body[:max_chars] + f"\n… [truncated; {total} chars total]"
    return f"{header}\n{body}"


def _coerce_indices(value: Any, *, max_indices: int) -> list[int]:
    """Coerce the `indices` argument to a bounded, de-duplicated list of ints.

    The `max_indices` cap is checked against the raw input length (so a
    pathological all-duplicate array is still rejected), then duplicates are
    dropped preserving first-occurrence order — recalling the same message
    twice would only waste the per-call budget and re-inject it twice.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"indices must be an array of integers, got {value!r}")
    if not value:
        raise ValueError("indices must not be empty")
    if len(value) > max_indices:
        raise ValueError(f"at most {max_indices} indices per call, got {len(value)}")
    coerced = [coerce_int("indices", item, minimum=0) for item in value]
    return list(dict.fromkeys(coerced))
