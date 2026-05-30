"""Context compression strategies.

The strategy contract (`CompressionStrategy` / `CompressionDecision`) lives
with `ContextPolicy` in `simple_agent_lab.context_view`; this module holds
the concrete strategies and the runtime that applies them, depending on
`context_view` one-way.

A strategy is a pure decision function. It looks at the messages the agent
currently sees and returns a `CompressionDecision`:

    return CompressionDecision(
        compress_indices=(...),   # which messages to remove from active view
        replacement=...,          # the single message that replaces them
    )

That's the entire strategy author surface. The framework handles everything
else: filtering `model_invisible_kinds`, validating that pinned messages
stay,
auto-fixing tool_call / tool_result pair splits, computing before/after
token totals, recording the replacement, and emitting the
`ContextCompressionEvent` that updates the active view and the trace.

Two built-in strategies ship by default:

- `ToolCompactStrategy` — rule-based, no LLM. Folds older tool-call /
  tool_result pairs into a single short marker. Cheap first stage.
- `SummarizeStrategy` — LLM-based. Asks a compressor model to produce a
  running summary of older droppable messages. More powerful fallback.

Compose them in `ContextPolicy.strategies` (evaluated in order) for a
tiered policy: try the cheap one first, fall back to the LLM only if
the context is still over budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from .context_view import (
    CompressionDecision,
    ContextPolicy,
    estimate_context_tokens,
)
from .messages import (
    AssistantMessage,
    Message,
    MessageKind,
    make_message,
    text_of,
    tool_results_of,
)
from .protocols import ContextCompressionEvent, Event

if TYPE_CHECKING:
    from .core import Agent
    from .state import State


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


@dataclass(frozen=True)
class ToolCompactStrategy:
    """Fold older tool-call/tool-result pairs into one compact marker.

    No LLM call — this is the cheap first stage of a tiered policy. When
    the active context exceeds `threshold_tokens`, every tool exchange
    except the most recent `keep_recent_exchanges` is collapsed into a
    single ``kind="summary"`` message that lists each call's tool name
    plus a short result preview (`preview_chars` characters).

    Typical setup with `SummarizeStrategy` as the LLM fallback:

        ContextPolicy(strategies=(
            ToolCompactStrategy(threshold_tokens=4000),
            SummarizeStrategy(
                compressor=summarizer,
                threshold_tokens=4000,
            ),
        ))
    """

    threshold_tokens: int
    keep_recent_exchanges: int = 1
    preview_chars: int = 200

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        if estimate_context_tokens([m for _, m in active]) <= self.threshold_tokens:
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
                "system",
                _format_compact_summary(active, old, self.preview_chars),
                sender="runtime",
                target=agent_name,
                kind="summary",
            ),
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
        before_tokens = estimate_context_tokens([message for _, message in active])
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
        output = self.compressor.generate(
            [message for _, message in to_compress] + [instruction]
        )
        summary_text = _output_text(output).strip() or (
            "Context was compressed, but the compressor returned no text."
        )
        return CompressionDecision(
            compress_indices=tuple(index for index, _ in to_compress),
            replacement=make_message(
                "system",
                summary_text,
                sender="runtime",
                target=agent_name,
                kind="summary",
            ),
        )


# ---------------------------------------------------------------------------
# Framework — strategy authors do not need to read past this line.
# ---------------------------------------------------------------------------


def maybe_compress_context(
    agent: "Agent",
    state: "State",
    policy: ContextPolicy,
) -> list[Event]:
    """Run each strategy in `policy.strategies` and apply its decision."""
    events: list[Event] = []
    for strategy in policy.strategies:
        active = [
            item for item in state.active_context_items() if policy.is_visible(item[1])
        ]
        decision = strategy(active, agent.name)
        if decision is None:
            continue
        events.extend(_apply_decision(agent, state, active, decision))
    return events


def _apply_decision(
    agent: "Agent",
    state: "State",
    active: list[tuple[int, Message]],
    decision: CompressionDecision,
) -> Iterator[Event]:
    compress_set = _align_tool_pairs(active, set(decision.compress_indices))
    if not compress_set:
        return

    # New active view: every uncompressed item stays in its original
    # order; the summary is spliced where the first compressed item was.
    # The framework stays agnostic to which kinds the strategy chose as
    # anchors — chronological order alone defines the result.
    first_compress_pos = next(
        (pos for pos, (index, _) in enumerate(active) if index in compress_set),
        len(active),
    )
    kept_before = [index for index, _ in active[:first_compress_pos]]
    kept_after = [
        index
        for index, _ in active[first_compress_pos + 1 :]
        if index not in compress_set
    ]
    kept_messages = [message for index, message in active if index not in compress_set]

    before_tokens = estimate_context_tokens([message for _, message in active])
    after_tokens = estimate_context_tokens(kept_messages + [decision.replacement])

    summary_event = state.record(decision.replacement)
    summary_index = len(state.messages) - 1
    compression_event = state.record_event(
        ContextCompressionEvent(
            agent=agent.name,
            summary_message_index=summary_index,
            compressed_message_indices=sorted(compress_set),
            active_context_indices=kept_before + [summary_index] + kept_after,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )
    )
    yield summary_event
    yield compression_event


def _align_tool_pairs(
    active: list[tuple[int, Message]],
    compress_set: set[int],
) -> set[int]:
    """Drop entries from `compress_set` whose tool partner is not compressed.

    A `tool_call` and its matching `tool_result` must travel together — many
    providers reject an orphan tool_result, and a tool_call with no result
    has no answer to reference. If a strategy compresses one side without
    the other, the framework un-compresses that side (conservative: keep
    more context rather than break the call/result graph).
    """
    by_index = {index: message for index, message in active}
    while True:
        orphans = {
            index
            for index in compress_set
            if any(
                partner not in compress_set
                for partner in _tool_partners(index, by_index)
            )
        }
        if not orphans:
            return compress_set
        compress_set = compress_set - orphans


def _tool_partners(index: int, by_index: dict[int, Message]) -> list[int]:
    """Indices of messages that pair with `index` via tool_call_id.

    Two cases, joined:
    - If `index` is an assistant with `tool_calls`, return every message
      that carries a matching `ToolResultBlock`.
    - If `index` carries `ToolResultBlock`s, return every assistant
      whose `tool_calls` they answer.
    """
    message = by_index[index]
    partners: list[int] = []
    if isinstance(message, AssistantMessage) and message.tool_calls:
        call_ids = {tool_call.id for tool_call in message.tool_calls}
        partners.extend(
            other_index
            for other_index, other_message in by_index.items()
            if other_index != index
            and any(
                block.tool_call_id in call_ids
                for block in tool_results_of(other_message.content)
            )
        )
    wanted = {block.tool_call_id for block in tool_results_of(message.content)}
    if wanted:
        partners.extend(
            other_index
            for other_index, other_message in by_index.items()
            if other_index != index
            and isinstance(other_message, AssistantMessage)
            and any(tool_call.id in wanted for tool_call in other_message.tool_calls)
        )
    return partners


def _output_text(message: Message) -> str:
    text = text_of(message.content)
    if text:
        return text
    return "\n".join(
        text_of(result.content) for result in tool_results_of(message.content)
    )
