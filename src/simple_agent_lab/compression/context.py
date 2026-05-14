"""Append-only context compression for long-horizon agent runs."""

from __future__ import annotations

from typing import Any, Protocol

from ..context_view import ContextPolicy, estimate_context_tokens
from ..messages import (
    AgentName,
    AssistantMessage,
    Message,
    make_message,
    text_of,
    tool_results_of,
)
from ..protocols import ContextCompressionEvent, Event, MessageEvent


class CompressionAgent(Protocol):
    name: AgentName
    step: Any


class CompressionState(Protocol):
    snapshot: Any

    @property
    def messages(self) -> list[Message]: ...

    def active_context_items(self) -> list[tuple[int, Message]]: ...

    def record(self, message: Message) -> MessageEvent: ...

    def context_compression(
        self,
        *,
        agent: AgentName,
        summary_message_index: int,
        compressed_message_indices: list[int],
        preserved_message_indices: list[int],
        recent_message_indices: list[int],
        before_tokens: int,
        after_tokens: int,
    ) -> ContextCompressionEvent: ...


def maybe_compress_context(
    agent: CompressionAgent,
    state: CompressionState,
    policy: ContextPolicy,
) -> list[Event]:
    threshold = policy.compress_at_tokens
    context_compressor = policy.compressor
    if context_compressor is None or threshold is None:
        return []

    active_items = [
        item
        for item in state.active_context_items()
        if item[1].kind not in policy.skip_kinds
    ]
    active_messages = [message for _, message in active_items]
    before_tokens = estimate_context_tokens(active_messages)
    if before_tokens <= threshold:
        return []

    preserved_items = [
        item for item in active_items if _preserve_during_compression(item[1])
    ]
    preserved_indices = [index for index, _ in preserved_items]
    preserved_index_set = set(preserved_indices)
    compressible_items = [
        item for item in active_items if item[0] not in preserved_index_set
    ]
    if len(compressible_items) <= policy.compress_keep_recent:
        return []

    if policy.compress_keep_recent:
        compress_items = compressible_items[: -policy.compress_keep_recent]
        recent_items = compressible_items[-policy.compress_keep_recent :]
    else:
        compress_items = list(compressible_items)
        recent_items = []
    compress_items, recent_items = _keep_tool_pair_boundary(
        compress_items,
        recent_items,
    )
    if not compress_items:
        return []

    prompt = _compression_prompt(agent.name, [message for _, message in compress_items])
    prompt_message = make_message(
        "user",
        prompt,
        sender="runtime",
        target=context_compressor.name,
        kind="task",
    )
    output = context_compressor.step(context_compressor, [prompt_message], state)
    summary = _full_message_text(output).strip()
    if not summary:
        summary = "Context was compressed, but the compressor returned no text."

    summary_message = make_message(
        "system",
        summary,
        sender="runtime",
        target=agent.name,
        kind="summary",
    )
    summary_event = state.record(summary_message)
    summary_message_index = len(state.snapshot.messages) - 1
    after_messages = [
        *(message for _, message in preserved_items),
        summary_message,
        *(message for _, message in recent_items),
    ]
    compression_event = state.context_compression(
        agent=agent.name,
        summary_message_index=summary_message_index,
        compressed_message_indices=[index for index, _ in compress_items],
        preserved_message_indices=preserved_indices,
        recent_message_indices=[index for index, _ in recent_items],
        before_tokens=before_tokens,
        after_tokens=estimate_context_tokens(after_messages),
    )
    return [summary_event, compression_event]


def _preserve_during_compression(message: Message) -> bool:
    return message.kind in {"task", "system"}


def _keep_tool_pair_boundary(
    compress_items: list[tuple[int, Message]],
    recent_items: list[tuple[int, Message]],
) -> tuple[list[tuple[int, Message]], list[tuple[int, Message]]]:
    while compress_items and recent_items:
        before = compress_items[-1][1]
        after = recent_items[0][1]
        if not _is_tool_result_for(after, before):
            break
        recent_items.insert(0, compress_items.pop())
    return compress_items, recent_items


def _is_tool_result_for(result: Message, assistant: Message) -> bool:
    if not isinstance(assistant, AssistantMessage) or not assistant.tool_calls:
        return False
    wanted = {tool_call.id for tool_call in assistant.tool_calls}
    return any(
        block.tool_call_id in wanted for block in tool_results_of(result.content)
    )


def _compression_prompt(agent_name: str, messages: list[Message]) -> str:
    rendered = "\n\n".join(
        _render_message_for_summary(index, message)
        for index, message in enumerate(messages, start=1)
    )
    return (
        f"Summarize the older conversation context for agent {agent_name!r}.\n"
        "Keep durable facts, decisions, tool results, constraints, and unresolved "
        "questions. Omit low-value wording. The summary will replace the messages "
        "below while the task and recent messages stay visible.\n\n"
        f"{rendered}"
    )


def _render_message_for_summary(index: int, message: Message) -> str:
    lines = [
        (
            f"{index}. role={message.role} sender={message.sender} "
            f"target={message.target} kind={message.kind}"
        )
    ]
    text = text_of(message.content).strip()
    if text:
        lines.append(text)
    if isinstance(message, AssistantMessage):
        for call in message.tool_calls:
            lines.append(
                f"[tool_call id={call.id} name={call.name} args={dict(call.arguments)!r}]"
            )
    for result in tool_results_of(message.content):
        result_text = text_of(result.content).strip()
        lines.append(
            f"[tool_result id={result.tool_call_id} name={result.tool_name}] "
            f"{result_text}"
        )
    return "\n".join(lines)


def _full_message_text(message: Message) -> str:
    text = text_of(message.content)
    if text:
        return text
    return "\n".join(
        text_of(result.content) for result in tool_results_of(message.content)
    )
