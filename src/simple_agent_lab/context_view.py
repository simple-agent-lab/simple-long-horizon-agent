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


# Default chars-per-token for the char-based fallback estimate. This is the
# *neutral* value measured empirically across content types (prose ~5.6, code
# ~3.9, json ~3.1, logs ~1.9 give an overall ~3.1 on mimo-v2.5-pro) — not a
# provider-accurate count. It is a default, not a constant: callers with a
# calibrated figure for their own model pass `chars_per_token=...` to the
# estimators. Re-measure per provider with `evals/compression/calibrate_tokens`.
CHARS_PER_TOKEN = 3.1
IMAGE_CHAR_ESTIMATE = 7373


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


def estimate_message_tokens(
    message: Message,
    *,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> int:
    """Best-effort per-message token estimate.

    Prefers the provider-reported `output_tokens` when an AssistantMessage
    carries one — that field is the exact tokenizer cost of this message and
    is stable across calls (so it's also the precise cost of re-sending it).
    For every other message and for assistants without usage data, falls back
    to the char/`chars_per_token` heuristic; pass a model-calibrated
    `chars_per_token` to override the default.

    Per-message attribution of `input_tokens` is not possible: providers only
    report that value as the SUM across all input messages of one call. So we
    do not split it backwards onto user/system/tool_result messages.
    """
    if isinstance(message, AssistantMessage):
        usage = message.usage
        if usage is not None and usage.output_tokens > 0:
            return usage.output_tokens
    chars = estimate_message_chars(message)
    return math.ceil(chars / chars_per_token)


def estimate_context_tokens(
    messages: Sequence[Message],
    *,
    allow_usage_baseline: bool = True,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> int:
    """Estimate context-window size for a selected message sequence.

    When possible, use the latest provider-reported assistant usage as the
    authoritative size of everything up to that response, then estimate only
    messages added after it. Pass ``allow_usage_baseline=False`` when the
    caller knows older context has been dropped or summarized — the historical
    usage figure would over-count text that is no longer present. Pass a
    model-calibrated ``chars_per_token`` to override the char-fallback default.
    """
    if allow_usage_baseline:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AssistantMessage) and _has_usage(message.usage):
                assert message.usage is not None
                return message.usage.context_tokens + sum(
                    estimate_message_tokens(trailing, chars_per_token=chars_per_token)
                    for trailing in messages[index + 1 :]
                )
    return sum(
        estimate_message_tokens(message, chars_per_token=chars_per_token)
        for message in messages
    )


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
