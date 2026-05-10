"""Model-visible context projection helpers.

`State.events` keeps the full trace. A context view is the smaller, explicit
projection an agent sees before one model step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    MessageChannel,
    MessageKind,
    SystemMessage,
    TextBlock,
    ToolResultMessage,
)


DEFAULT_SKIP_KINDS: tuple[MessageKind, ...] = ("notification", "trace")
DEFAULT_PINNED_KINDS: tuple[MessageKind, ...] = ("task", "system", "summary")

# Same teaching-level heuristic as the runtime token estimators: this is a
# stable budget signal, not provider-accurate token accounting.
CHARS_PER_TOKEN = 4
IMAGE_CHAR_ESTIMATE = 7373


@dataclass(frozen=True)
class ContextPolicy:
    """Visibility and budget policy for one agent context view."""

    include_broadcast: bool = True
    include_self: bool = True
    include_system: bool = True
    channels: tuple[MessageChannel, ...] | None = None
    skip_kinds: tuple[MessageKind, ...] = DEFAULT_SKIP_KINDS
    pinned_kinds: tuple[MessageKind, ...] = DEFAULT_PINNED_KINDS
    last: int | None = None
    max_chars: int | None = None
    max_message_chars: int | None = None
    reserve_recent: int = 1

    def __post_init__(self) -> None:
        _validate_positive("last", self.last)
        _validate_positive("max_chars", self.max_chars)
        _validate_positive("max_message_chars", self.max_message_chars)
        if self.reserve_recent < 0:
            raise ValueError(
                f"ContextPolicy.reserve_recent must be >= 0, got {self.reserve_recent!r}"
            )


@dataclass(frozen=True)
class ContextStats:
    total_messages: int
    visible_messages: int
    selected_messages: int
    dropped_messages: int
    estimated_chars: int
    estimated_tokens: int
    clipped_messages: int
    usage_known_messages: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total_messages": self.total_messages,
            "visible_messages": self.visible_messages,
            "selected_messages": self.selected_messages,
            "dropped_messages": self.dropped_messages,
            "estimated_chars": self.estimated_chars,
            "estimated_tokens": self.estimated_tokens,
            "clipped_messages": self.clipped_messages,
            "usage_known_messages": self.usage_known_messages,
        }


@dataclass(frozen=True)
class ContextView:
    agent: str
    messages: tuple[Message, ...]
    stats: ContextStats
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            **self.stats.as_dict(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _ContextGroup:
    messages: tuple[Message, ...]
    chars: int
    pinned: bool


def build_context_view(
    agent_name: str,
    messages: Sequence[Message],
    *,
    policy: ContextPolicy | None = None,
) -> ContextView:
    """Project a full transcript into the messages visible to one agent."""
    resolved = policy or ContextPolicy()
    visible = [
        message
        for message in messages
        if is_visible_to_agent(message, agent_name, resolved)
    ]
    if resolved.last is not None:
        visible = visible[-resolved.last :]

    clipped_messages = 0
    clipped: list[Message] = []
    for message in visible:
        next_message, did_clip = clip_message(message, resolved.max_message_chars)
        clipped.append(next_message)
        if did_clip:
            clipped_messages += 1

    if resolved.max_chars is None:
        # No budget → no grouping or per-group char accounting needed.
        selected_messages = tuple(clipped)
        dropped_count = 0
        budget_notes: list[str] = []
        estimated_chars = sum(estimate_message_chars(message) for message in clipped)
    else:
        groups = _group_messages(clipped, resolved)
        selected_groups, dropped_groups, budget_notes = _select_groups(groups, resolved)
        selected_messages = tuple(
            message for group in selected_groups for message in group.messages
        )
        dropped_count = sum(len(group.messages) for group in dropped_groups)
        estimated_chars = sum(group.chars for group in selected_groups)
    estimated_tokens = sum(estimate_message_tokens(message) for message in selected_messages)
    usage_known_messages = sum(
        1
        for message in selected_messages
        if isinstance(message, AssistantMessage)
        and message.usage is not None
        and message.usage.output_tokens > 0
    )
    stats = ContextStats(
        total_messages=len(messages),
        visible_messages=len(visible),
        selected_messages=len(selected_messages),
        dropped_messages=dropped_count,
        estimated_chars=estimated_chars,
        estimated_tokens=estimated_tokens,
        clipped_messages=clipped_messages,
        usage_known_messages=usage_known_messages,
    )
    return ContextView(
        agent=agent_name,
        messages=selected_messages,
        stats=stats,
        notes=tuple(budget_notes),
    )


def is_visible_to_agent(
    message: Message,
    agent_name: str,
    policy: ContextPolicy | None = None,
) -> bool:
    """Return whether a runtime message belongs in an agent's candidate view."""
    resolved = policy or ContextPolicy()
    if message.kind in resolved.skip_kinds:
        return False
    if resolved.channels is not None and message.channel not in resolved.channels:
        return False
    if isinstance(message, SystemMessage) and not resolved.include_system:
        return False
    if message.target == agent_name:
        return True
    if resolved.include_broadcast and message.target == "all":
        return True
    if resolved.include_self and message.sender == agent_name:
        return True
    return False


def estimate_message_chars(message: Message) -> int:
    """Estimate model-visible character cost for a runtime message."""
    meta_chars = sum(
        len(str(value))
        for value in (
            message.role,
            message.sender,
            message.target,
            message.kind,
            message.channel,
        )
    )
    content_chars = _content_chars(message.content)
    if isinstance(message, AssistantMessage):
        content_chars += sum(len(thinking.text) for thinking in message.thinking)
        content_chars += sum(
            len(call.id) + len(call.name) + len(repr(dict(call.arguments)))
            for call in message.tool_calls
        )
    if isinstance(message, ToolResultMessage):
        content_chars += len(message.tool_call_id) + len(message.tool_name)
    return meta_chars + content_chars


def estimate_message_tokens(message: Message) -> int:
    """Best-effort per-message token estimate.

    Prefers the provider-reported `output_tokens` when an AssistantMessage
    carries one — that field is the exact tokenizer cost of this message and
    is stable across calls (so it's also the precise cost of re-sending it).
    For every other message and for assistants without usage data, falls back
    to the char/4 heuristic.

    Per-message attribution of `input_tokens` is not possible: providers only
    report that value as the SUM across all input messages of one call. So we
    do not split it backwards onto user/system/tool_result messages.
    """
    if isinstance(message, AssistantMessage):
        usage = message.usage
        if usage is not None and usage.output_tokens > 0:
            return usage.output_tokens
    chars = estimate_message_chars(message)
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def clip_message(
    message: Message,
    max_chars: int | None,
) -> tuple[Message, bool]:
    """Clip large text content while preserving message identity fields."""
    if max_chars is None:
        return message, False
    content = message.content
    if not isinstance(content, str) or len(content) <= max_chars:
        return message, False

    marker = f"\n[context clipped: omitted {len(content) - max_chars} chars]"
    keep_chars = max(0, max_chars - len(marker))
    marker = f"\n[context clipped: omitted {len(content) - keep_chars} chars]"
    return replace(message, content=f"{content[:keep_chars]}{marker}"), True


def _group_messages(
    messages: Sequence[Message],
    policy: ContextPolicy,
) -> list[_ContextGroup]:
    groups: list[_ContextGroup] = []
    consumed: set[int] = set()
    for index, message in enumerate(messages):
        if index in consumed:
            continue
        group = [message]
        if isinstance(message, AssistantMessage) and message.tool_calls:
            wanted = {tool_call.id for tool_call in message.tool_calls}
            next_index = index + 1
            while next_index < len(messages) and wanted:
                candidate = messages[next_index]
                if (
                    isinstance(candidate, ToolResultMessage)
                    and candidate.tool_call_id in wanted
                ):
                    group.append(candidate)
                    consumed.add(next_index)
                    wanted.remove(candidate.tool_call_id)
                    next_index += 1
                    continue
                break
        groups.append(
            _ContextGroup(
                messages=tuple(group),
                chars=sum(estimate_message_chars(item) for item in group),
                pinned=any(_is_pinned(item, policy) for item in group),
            )
        )
    return groups


def _select_groups(
    groups: Sequence[_ContextGroup],
    policy: ContextPolicy,
) -> tuple[list[_ContextGroup], list[_ContextGroup], list[str]]:
    if policy.max_chars is None:
        return list(groups), [], []

    max_chars = policy.max_chars
    selected: set[int] = set()
    notes: list[str] = []
    total_chars = 0

    def add(index: int, *, force: bool) -> bool:
        nonlocal total_chars
        if index in selected:
            return True
        group = groups[index]
        if force or total_chars + group.chars <= max_chars:
            selected.add(index)
            total_chars += group.chars
            return True
        return False

    for index, group in enumerate(groups):
        if group.pinned:
            add(index, force=True)

    if policy.reserve_recent:
        first_recent = max(0, len(groups) - policy.reserve_recent)
        for index in range(first_recent, len(groups)):
            add(index, force=True)

    for index in range(len(groups) - 1, -1, -1):
        add(index, force=False)

    selected_groups = [group for index, group in enumerate(groups) if index in selected]
    dropped_groups = [group for index, group in enumerate(groups) if index not in selected]
    dropped_count = sum(len(group.messages) for group in dropped_groups)
    if dropped_count:
        notes.append(f"budget dropped {dropped_count} message(s)")
    if total_chars > max_chars:
        notes.append("pinned/recent context exceeds max_chars")
    return selected_groups, dropped_groups, notes


def _is_pinned(message: Message, policy: ContextPolicy) -> bool:
    return isinstance(message, SystemMessage) or message.kind in policy.pinned_kinds


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


def _validate_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"ContextPolicy.{name} must be > 0, got {value!r}")
