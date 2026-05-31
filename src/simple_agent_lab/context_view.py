"""Model-visible context projection: policy, contract, and view builder.

`State.events` keeps the full trace. A context view is the smaller, explicit
projection an agent sees before one model step. Projection here means only
visibility filtering: messages whose `kind` is in `model_invisible_kinds`
are dropped because they were never meant for the model.

This module owns the two pieces that describe *what* should happen to a
view: `ContextPolicy` (the per-agent config) and the compression contract
(`CompressionStrategy` / `CompressionDecision`) that policies reference.
The concrete strategies and the runtime that applies them live in
`simple_agent_lab.compression`, which depends on this module — never the
other way around. Strategies mutate the active view through
`ContextCompressionEvent` *before* `build_context_view` runs, so this module
never has to know how the active view was shaped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    MessageKind,
    TextBlock,
    TokenUsage,
    tool_results_of,
)


# Chars-per-token for the char-based fallback estimate — the *neutral* value
# measured empirically across content types (prose ~5.6, code ~3.9, json ~3.1,
# logs ~1.9 give an overall ~3.1 on mimo-v2.5-pro), not a provider-accurate
# count. One global ratio is deliberately enough: the estimate only covers the
# small tail since the last provider usage report, and the runtime's safety
# buffer absorbs the residual error, so per-provider calibration is not worth
# the complexity (see ADR 0017).
CHARS_PER_TOKEN = 3.1
IMAGE_CHAR_ESTIMATE = 7373

# Default reserves for the effective context budget (mirrors opencode's
# OUTPUT_TOKEN_MAX / COMPACTION_BUFFER). They assume a large window — tune per
# model. The safety buffer matters because the size signal partly rests on the
# char estimate, which is approximate; the buffer is the slack that keeps that
# error from overflowing the real window.
DEFAULT_OUTPUT_RESERVE = 32000
DEFAULT_SAFETY_BUFFER = 20000


@dataclass(frozen=True)
class CompressionDecision:
    """What a strategy returns.

    `compress_indices` lists the positions in `state.messages` that should
    be removed from the active view. `replacement` is the single message
    the framework writes in their place (typically a `kind="summary"`
    system message).
    """

    compress_indices: tuple[int, ...]
    replacement: Message


class CompressionStrategy(Protocol):
    """Signature every strategy must satisfy.

    Args:
        active: `(index, message)` pairs the agent currently sees, in
            display order. Already filtered to exclude
            `policy.model_invisible_kinds`, so the strategy can act on
            every item it receives.
        agent_name: target for the replacement message (typically the
            agent whose context is being shrunk).

    A strategy is free to pick any subset of indices for compression.
    Convention is that each strategy owns a `preserve_kinds` knob so the
    caller can tune what stays anchored (see
    `simple_agent_lab.compression.DEFAULT_PRESERVE_KINDS`). Concrete
    strategies live in `simple_agent_lab.compression`.
    """

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None: ...


@dataclass(frozen=True)
class ContextPolicy:
    """Visibility filter + compression strategy list for one agent.

    `model_invisible_kinds` are never shown to the model. It defaults to
    empty (every runtime `kind` is model-visible); set it per agent to hide
    kinds that your own extension code records but should not project to the
    LLM. `strategies` is evaluated in order before each model request; each
    strategy may return a `CompressionDecision` that the runtime applies
    to state.

    `model_invisible_kinds` is *visibility*, not *compression*. It decides
    whether a kind reaches the model at all — it is not the way to keep a
    kind out of summaries. To exempt a kind from compression (e.g. keep
    `"system"` or `"task"` verbatim instead of folding it into a summary),
    use a strategy's `preserve_kinds` knob (see
    `simple_agent_lab.compression.DEFAULT_PRESERVE_KINDS`, which already
    pins `task`/`system`/`summary`/`context`). Do not add `"system"` here
    to protect it from compression — that would hide the system messages
    from the model entirely.
    """

    model_invisible_kinds: tuple[MessageKind, ...] = ()
    strategies: tuple[CompressionStrategy, ...] = field(default_factory=tuple)

    def is_visible(self, message: Message) -> bool:
        """Whether this message survives the agent's visibility filter."""
        return message.kind not in self.model_invisible_kinds


@dataclass(frozen=True)
class ContextStats:
    total_messages: int
    visible_messages: int
    estimated_chars: int
    estimated_tokens: int
    usage_known_messages: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total_messages": self.total_messages,
            "visible_messages": self.visible_messages,
            "estimated_chars": self.estimated_chars,
            "estimated_tokens": self.estimated_tokens,
            "usage_known_messages": self.usage_known_messages,
        }


@dataclass(frozen=True)
class ContextView:
    agent: str
    messages: tuple[Message, ...]
    stats: ContextStats

    def as_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            **self.stats.as_dict(),
        }


def build_context_view(
    agent_name: str,
    messages: Sequence[Message],
    *,
    policy: ContextPolicy | None = None,
) -> ContextView:
    """Project a transcript into the messages visible to one agent.

    This is a pure visibility filter. To shrink the active context, attach
    a `CompressionStrategy` to `ContextPolicy.strategies`; the runtime
    runs strategies before this call and `messages` will already reflect
    their effect.
    """
    resolved = policy or ContextPolicy()
    visible = tuple(message for message in messages if resolved.is_visible(message))
    estimated_chars = sum(estimate_message_chars(message) for message in visible)
    estimated_tokens = estimate_context_tokens(visible)
    usage_known_messages = sum(
        1
        for message in visible
        if isinstance(message, AssistantMessage)
        and message.usage is not None
        and message.usage.output_tokens > 0
    )
    stats = ContextStats(
        total_messages=len(messages),
        visible_messages=len(visible),
        estimated_chars=estimated_chars,
        estimated_tokens=estimated_tokens,
        usage_known_messages=usage_known_messages,
    )
    return ContextView(agent=agent_name, messages=visible, stats=stats)


def estimate_message_chars(message: Message) -> int:
    """Estimate model-visible character cost for a runtime message."""
    meta_chars = sum(
        len(str(value))
        for value in (
            message.role,
            message.sender,
            message.target,
            message.kind,
        )
    )
    content_chars = _content_chars(message.content)
    if isinstance(message, AssistantMessage):
        content_chars += sum(len(thinking.text) for thinking in message.thinking)
        content_chars += sum(
            len(call.id) + len(call.name) + len(repr(dict(call.arguments)))
            for call in message.tool_calls
        )
    for block in tool_results_of(message.content):
        content_chars += len(block.tool_call_id) + len(block.tool_name)
    return meta_chars + content_chars


def estimate_message_tokens(message: Message) -> int:
    """Best-effort per-message token estimate.

    Prefers the provider-reported `output_tokens` when an AssistantMessage
    carries one — that field is the exact tokenizer cost of this message and
    is stable across calls (so it's also the precise cost of re-sending it).
    For every other message and for assistants without usage data, falls back
    to the char/`CHARS_PER_TOKEN` heuristic.

    Per-message attribution of `input_tokens` is not possible: providers only
    report that value as the SUM across all input messages of one call. So we
    do not split it backwards onto user/system/tool_result messages.
    """
    if isinstance(message, AssistantMessage):
        usage = message.usage
        if usage is not None and usage.output_tokens > 0:
            return usage.output_tokens
    chars = estimate_message_chars(message)
    return math.ceil(chars / CHARS_PER_TOKEN)


def estimate_context_tokens(
    messages: Sequence[Message],
    *,
    allow_usage_baseline: bool = True,
) -> int:
    """Estimate context-window size for a selected message sequence.

    When possible, take the latest provider-reported assistant usage as the
    authoritative size of everything up to that response, and estimate only the
    messages added after it — keeping the inaccurate char estimate confined to
    the smallest possible tail (the runtime's safety buffer covers that bit).
    Pass ``allow_usage_baseline=False`` when older context has been dropped or
    summarized — the historical usage figure would over-count text that is no
    longer present, so the whole sequence is estimated instead.
    """
    if allow_usage_baseline:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AssistantMessage) and _has_usage(message.usage):
                assert message.usage is not None
                return message.usage.context_tokens + sum(
                    estimate_message_tokens(trailing)
                    for trailing in messages[index + 1 :]
                )
    return sum(estimate_message_tokens(message) for message in messages)


def effective_token_budget(
    context_window: int,
    *,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    safety_buffer: int = DEFAULT_SAFETY_BUFFER,
) -> int:
    """Usable input budget = window − reserved output − safety buffer.

    Compression should trigger against this, not the raw `context_window`.
    Two reasons to hold tokens back:

    - `output_reserve`: the model still needs room to generate its reply, so
      that much of the window is never available for input.
    - `safety_buffer`: part of the size signal is the char estimate, which is
      approximate. The buffer is the slack that keeps that estimate's error
      (and any not-yet-reported trailing message) from overflowing the real
      window. The less the size is provider-confirmed, the larger it should be.
    """
    return max(0, context_window - output_reserve - safety_buffer)


def _content_chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, tuple):
        total = 0
        for block in content:
            if isinstance(block, TextBlock):
                total += len(block.text)
            elif isinstance(block, ImageBlock):
                total += IMAGE_CHAR_ESTIMATE
        return total
    return len(str(content))


def _has_usage(usage: TokenUsage | None) -> bool:
    return usage is not None and usage.context_tokens > 0
