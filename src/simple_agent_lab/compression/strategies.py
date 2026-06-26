"""Concrete compression strategies — the strategy-author surface.

A strategy is a pure decision function `(active, agent_name) -> decision`.
It looks at the messages the agent currently sees and returns a
`CompressionDecision`:

    return CompressionDecision(
        compress_indices=(...),   # which messages to remove from active view
        replacement=...,          # the single message that replaces them
    )

That's the entire surface. The `runtime` module handles everything else:
filtering `model_invisible_kinds`, validating pinned messages, auto-fixing
tool_call / tool_result pair splits, sizing, recording the replacement, and
emitting the `ContextCompressionEvent`.

Setting `rewrite=True` switches the decision from this N->1 fold to a 1->1,
in-place substitution: `compress_indices` names exactly one message and
`replacement` supersedes it while preserving its role and `tool_call_id`
linkage (so tool pairs stay intact without alignment). Use it to shrink one
message (e.g. truncate a large `tool_result`) rather than collapse a whole
exchange into a summary.

Two strategies ship, as swappable alternatives — set `ContextPolicy.strategy`
to one of them:

- `ToolCompactStrategy` — rule-based, no LLM. Folds older tool-call /
  tool_result pairs into a single short marker. Cheap.
- `SummarizeStrategy` — LLM-based. Asks a compressor model to produce a
  running summary of older droppable messages. More powerful.

`TieredStrategy` combines several into one (first-applicable wins), so a
cheap-fold-then-LLM-summary tier is itself a single strategy you drop into the
one `strategy` slot.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..context_view import (
    CompressionDecision,
    CompressionStrategy,
    build_context_view,
)
from ..llm.bridge import messages_to_llm_messages
from ..llm.types import llm_message
from ..messages import (
    AssistantMessage,
    Message,
    MessageSidecar,
    MessageKind,
    make_message,
    text_of,
    tool_calls_of,
    tool_results_of,
)
from ..protocols import ModelRequestEvent, ModelResponseEvent
from .runtime import _active_context_tokens, _tool_partners

if TYPE_CHECKING:
    from ..core import Agent


# Conventional default for strategies that want a safe "don't drop these"
# list. Strategies are free to override (e.g. a cascading-summarize
# strategy that wants to fold prior summaries can pass
# `preserve_kinds=("task", "system", "context")` and let `"summary"`
# fall into the compressible pool).
#
# - `task`:    the original instruction that started the run
# - `system`:  runtime/agent-level policy
# - `summary`: output of a prior compression round
# - `context`: sub-agent operating context recorded by `task_tool`
DEFAULT_PRESERVE_KINDS: tuple[MessageKind, ...] = (
    "task",
    "system",
    "summary",
    "context",
)


def format_index_ranges(indices: Sequence[int]) -> str:
    """Render indices as compact sorted ranges: (2, 3, 4, 7) -> '2-4, 7'."""
    ordered = sorted(set(indices))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for index in ordered[1:]:
        if index == prev + 1:
            prev = index
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = index
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


def source_note(indices: Sequence[int]) -> str:
    """Provenance footer naming the transcript messages a replacement folded.

    `State` is append-only, so compression never deletes the originals — it
    only removes them from the active view. This line tells the model where
    they live; the `recall` tool (`simple_agent_lab.tools.recall`) reads the
    citation back and fetches the originals by index. Summaries cite, recall
    retrieves: compression becomes recoverable instead of lossy.
    """
    return (
        f"[Compressed from transcript messages {format_index_ranges(indices)}. "
        "Originals are retrievable by index via the `recall` tool when it is "
        "available.]"
    )


@dataclass(frozen=True)
class ToolCompactStrategy:
    """Fold older tool-call/tool-result pairs into one compact marker.

    No LLM call — the cheap, rule-based option. When the active context
    exceeds `threshold_tokens`, every tool exchange except the most recent
    `keep_recent_exchanges` is collapsed into a single ``kind="summary"``
    message that lists each call's tool name plus a short result preview
    (`preview_chars` characters).

        ContextPolicy(strategy=ToolCompactStrategy(threshold_tokens=4000))
    """

    threshold_tokens: int
    keep_recent_exchanges: int = 1
    preview_chars: int = 200

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        if _active_context_tokens(active) <= self.threshold_tokens:
            return None
        exchanges = [
            exchange for exchange in _find_tool_exchanges(active) if exchange[1]
        ]
        if len(exchanges) <= self.keep_recent_exchanges:
            return None
        old = (
            exchanges[: -self.keep_recent_exchanges]
            if self.keep_recent_exchanges
            else exchanges
        )
        compress_indices = tuple(
            index
            for assistant_index, result_indices in old
            for index in (assistant_index, *result_indices)
        )
        return CompressionDecision(
            compress_indices=compress_indices,
            replacement=make_message(
                "user",
                _format_compact_summary(active, old, self.preview_chars)
                + "\n"
                + source_note(compress_indices),
                sender="runtime",
                target=agent_name,
                kind="summary",
            ),
            label="tool-compact",
        )


def _find_tool_exchanges(
    active: list[tuple[int, Message]],
) -> list[tuple[int, tuple[int, ...]]]:
    """Pair every assistant tool-call message with its tool_result partners.

    Order follows the assistants' original order in `active`. Result
    pairing reuses `_tool_partners` so this stays in sync with the
    framework's split-pair auto-fix.
    """
    by_index = {index: message for index, message in active}
    return [
        (index, tuple(_tool_partners(index, by_index)))
        for index, message in active
        if isinstance(message, AssistantMessage) and message.tool_calls
    ]


def _format_compact_summary(
    active: list[tuple[int, Message]],
    exchanges: list[tuple[int, tuple[int, ...]]],
    preview_chars: int,
) -> str:
    by_index = {index: message for index, message in active}
    lines = [f"Compacted {len(exchanges)} older tool exchange(s):"]
    for assistant_index, result_indices in exchanges:
        assistant = by_index[assistant_index]
        if not isinstance(assistant, AssistantMessage):
            continue
        for tool_call in assistant.tool_calls:
            result_text = _find_tool_result_text(by_index, result_indices, tool_call.id)
            preview = (
                result_text[:preview_chars] + "…"
                if len(result_text) > preview_chars
                else result_text
            )
            lines.append(f"- {tool_call.name}: {preview!r}")
    return "\n".join(lines)


def _find_tool_result_text(
    by_index: dict[int, Message],
    result_indices: tuple[int, ...],
    tool_call_id: str,
) -> str:
    for result_index in result_indices:
        for block in tool_results_of(by_index[result_index].content):
            if block.tool_call_id == tool_call_id:
                return text_of(block.content)
    return ""


@dataclass(frozen=True)
class SummarizeStrategy:
    """Fold older messages into one LLM-generated summary.

    The compressor is invoked as a normal LLM call: older messages plus
    one trailing instruction. Its output text becomes the replacement.

    `preserve_kinds` declares which kinds stay verbatim and never go
    into the summary input. Default keeps prior summaries verbatim;
    set `preserve_kinds=("task", "system", "context")` to enable
    cascading summarization (older summaries get folded into the new
    one).
    """

    compressor: "Agent"
    threshold_tokens: int
    keep_recent: int = 4
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        if not active:
            return None
        before_tokens = _active_context_tokens(active)
        if before_tokens <= self.threshold_tokens:
            return None
        droppable = [item for item in active if item[1].kind not in self.preserve_kinds]
        if len(droppable) <= self.keep_recent:
            return None
        to_compress = droppable[: -self.keep_recent] if self.keep_recent else droppable
        if not to_compress:
            return None

        instruction = make_message(
            "user",
            (
                f"Summarize the older conversation context above for agent "
                f"{agent_name!r}.\n"
                "Keep durable facts, decisions, tool results, constraints, "
                "and unresolved questions. Omit low-value wording. Your "
                "summary will replace the prior messages while the task and "
                "recent messages stay visible."
            ),
            sender="runtime",
            target=self.compressor.name,
            kind="task",
        )
        compressor_messages = [message for _, message in to_compress] + [instruction]
        compressor_context = build_context_view(
            self.compressor.name,
            compressor_messages,
        )
        llm_payload = messages_to_llm_messages(
            list(compressor_context.messages),
            with_header=False,
        )
        if self.compressor.system_prompt:
            llm_payload = [
                llm_message("system", self.compressor.system_prompt),
                *llm_payload,
            ]
        record_model_events = _agent_records_model_events(self.compressor)
        trace_events: tuple[ModelRequestEvent | ModelResponseEvent, ...] = ()
        if record_model_events:
            request_event = ModelRequestEvent(
                agent=self.compressor.name,
                visible_count=len(compressor_context.messages),
                llm_message_count=len(llm_payload),
                context_view=compressor_context.as_dict(),
                tools=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in self.compressor.tools
                ],
                llm_payload=llm_payload,
            )
        started = time.monotonic()
        output = self.compressor.generate(compressor_messages)
        elapsed = time.monotonic() - started
        if record_model_events:
            response_event = ModelResponseEvent(
                agent=self.compressor.name,
                output_kind=output.kind,
                target=output.target,
                tool_call_count=len(tool_calls_of(output.content)),
                usage=output.usage if isinstance(output, AssistantMessage) else None,
                model=output.model if isinstance(output, AssistantMessage) else "",
                elapsed=elapsed,
            )
            trace_events = (request_event, response_event)
        summary_text = _output_text(output).strip() or (
            "Context was compressed, but the compressor returned no text."
        )
        compress_indices = tuple(index for index, _ in to_compress)
        return CompressionDecision(
            compress_indices=compress_indices,
            replacement=make_message(
                "user",
                summary_text + "\n\n" + source_note(compress_indices),
                sender="runtime",
                target=agent_name,
                kind="summary",
                sidecar=_compression_sidecar(output),
            ),
            label="summarize",
            trace_events=trace_events,
        )


def _agent_records_model_events(agent: "Agent") -> bool:
    provider = getattr(agent, "llm_provider", None)
    return provider is not None and provider.api != "fake"


def _compression_sidecar(output: Message) -> MessageSidecar:
    sidecar: MessageSidecar = {}
    raw = output.sidecar.get("raw")
    if raw:
        sidecar["raw"] = raw
    if isinstance(output, AssistantMessage):
        metadata: dict[str, object] = {"compressor": output.sender}
        if output.model:
            metadata["model"] = output.model
        if output.usage is not None:
            metadata["usage"] = {
                "input_tokens": output.usage.input_tokens,
                "output_tokens": output.usage.output_tokens,
                "cache_read_tokens": output.usage.cache_read_tokens,
                "cache_write_tokens": output.usage.cache_write_tokens,
            }
        sidecar["compression"] = metadata
    return sidecar


def _output_text(message: Message) -> str:
    text = text_of(message.content)
    if text:
        return text
    return "\n".join(
        text_of(result.content) for result in tool_results_of(message.content)
    )


@dataclass(frozen=True)
class TieredStrategy:
    """Combine several strategies into one — first applicable wins.

    The policy holds a single `strategy`; this composite lets you express a
    tiered policy without bringing a list back to `ContextPolicy`. Its `stages`
    are tried in order each pass and the first to return a `CompressionDecision`
    wins — e.g. a cheap rule-based fold first, an LLM summary only when the fold
    has nothing left to do:

        ContextPolicy(strategy=TieredStrategy((
            ToolCompactStrategy(threshold_tokens=4000),
            SummarizeStrategy(compressor=summarizer, threshold_tokens=4000),
        )))

    One decision is applied per compression pass (per model request); across
    turns the stages keep firing as the context grows. Applying more than one
    stage within a single pass would be a runtime choice, not this strategy's.
    """

    stages: tuple[CompressionStrategy, ...]

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        for stage in self.stages:
            decision = stage(active, agent_name)
            if decision is not None:
                return decision
        return None
