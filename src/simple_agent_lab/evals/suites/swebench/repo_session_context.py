"""Shared repo-session context repair helpers.

SWE-bench Pro repo sessions run one instance after another while carrying
active context forward. These helpers keep invalid-prompt and tool-pair context
surgery in one place so host tests and the in-container runner exercise the same
behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from simple_agent_lab.messages import (
    is_tool_result_message,
    message_tool_calls,
    tool_results_of,
    user_message,
)
from simple_agent_lab.protocols import ContextCompressionEvent

INVALID_PROMPT_TOOL_REMINDER = (
    "刚刚的工具调用及其输出会触发 invalid_prompt，已从上下文移除。请使用其他命令继续。"
)
INVALID_PROMPT_INSTANCE_END_MESSAGE = (
    "上一道题 {instance_id} 在这里结束；因为工具输出持续触发 invalid_prompt，"
    "已跳过该实例。继续下一道题。"
)
INVALID_PROMPT_TOOL_RETRY_LIMIT = 20
InvalidPromptSource = Literal["instance_task", "tool_output", "unknown"]


def is_invalid_prompt_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    return (
        "invalid_prompt" in text
        or "invalid prompt" in text
        or "-4321" in text
        or code == -4321
        or status_code == -4321
    )


def invalid_prompt_source(state: Any, *, instance_id: str) -> InvalidPromptSource:
    """Classify which latest user-visible message caused invalid_prompt."""

    for _, message in reversed(state.active_context_items()):
        if getattr(message, "role", "") != "user":
            continue
        if is_tool_result_message(message):
            return "tool_output"
        if message_swebench_instance_id(message) == instance_id:
            return "instance_task"
        return "unknown"
    return "unknown"


def replace_latest_tool_exchange_for_invalid_prompt(
    state: Any, *, agent_name: str
) -> bool:
    """Replace the latest active tool call/result exchange with a safe note."""

    active_items = state.active_context_items()
    tool_result_index: int | None = None
    tool_call_ids: set[str] = set()
    for index, message in reversed(active_items):
        if is_tool_result_message(message):
            tool_result_index = index
            tool_call_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            break
    if tool_result_index is None:
        return False

    dropped = tool_exchange_indices(active_items, tool_call_ids)
    if not dropped:
        return False

    replacement = user_message(
        INVALID_PROMPT_TOOL_REMINDER,
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(replacement)
    replacement_index = len(state.messages) - 1
    active_context_indices: list[int] = []
    inserted = False
    for index, _ in active_items:
        if index in dropped:
            if not inserted:
                active_context_indices.append(replacement_index)
                inserted = True
            continue
        active_context_indices.append(index)
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=replacement_index,
            compressed_message_indices=sorted(dropped),
            active_context_indices=active_context_indices,
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-tool-exchange-replace",
        )
    )
    return True


def tool_exchange_indices(
    active_items: list[tuple[int, Any]], tool_call_ids: set[str]
) -> set[int]:
    """Return the connected tool-call/tool-result component for call ids."""

    wanted = set(tool_call_ids)
    dropped: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, message in active_items:
            calls = message_tool_calls(message)
            result_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            if calls and any(call.id in wanted for call in calls):
                before = len(wanted)
                wanted.update(call.id for call in calls)
                dropped.add(index)
                changed = changed or len(wanted) != before
            if result_ids and result_ids & wanted:
                before = len(wanted)
                wanted.update(result_ids)
                dropped.add(index)
                changed = changed or len(wanted) != before
    return dropped


def repair_active_tool_pairs(state: Any, *, agent_name: str) -> bool:
    """Drop active tool-call/result orphans before the next provider request."""

    active_items = state.active_context_items()
    kept = tool_pair_safe_indices(active_items)
    if len(kept) == len(active_items):
        return False

    dropped = [index for index, _ in active_items if index not in set(kept)]
    note = user_message(
        "Removed an incomplete tool call/tool result exchange from context.",
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(note)
    note_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=note_index,
            compressed_message_indices=dropped,
            active_context_indices=[*kept, note_index],
            before_tokens=0,
            after_tokens=0,
            strategy="tool-pair-orphan-repair",
        )
    )
    return True


def tool_pair_safe_indices(active_items: list[tuple[int, Any]]) -> list[int]:
    remaining = {index for index, _ in active_items}
    messages = dict(active_items)
    changed = True
    while changed:
        changed = False
        call_ids = {
            call.id
            for index in remaining
            for call in message_tool_calls(messages[index])
        }
        result_ids = {
            block.tool_call_id
            for index in remaining
            for block in tool_results_of(messages[index].content)
        }
        drop: set[int] = set()
        for index in remaining:
            calls = message_tool_calls(messages[index])
            if calls and any(call.id not in result_ids for call in calls):
                drop.add(index)
            results = tool_results_of(messages[index].content)
            if results and any(block.tool_call_id not in call_ids for block in results):
                drop.add(index)
        if drop:
            remaining -= drop
            changed = True
    return [index for index, _ in active_items if index in remaining]


def drop_instance_task_for_invalid_prompt_skip(
    state: Any, *, agent_name: str, instance_id: str
) -> bool:
    """Drop a skipped problem statement from active context."""

    active_items = state.active_context_items()
    target_index: int | None = None
    for index, message in reversed(active_items):
        if (
            getattr(message, "role", "") == "user"
            and message_swebench_instance_id(message) == instance_id
        ):
            target_index = index
            break
    if target_index is None:
        return False

    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=target_index,
            compressed_message_indices=[target_index],
            active_context_indices=[
                index for index, _ in active_items if index != target_index
            ],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-instance-task-drop",
        )
    )
    return True


def end_instance_after_invalid_prompt_tool_retry_limit(
    state: Any, *, agent_name: str, instance_id: str
) -> bool:
    """Clear active context after persistent invalid_prompt for one instance."""

    active_items = state.active_context_items()
    if not active_items:
        return False

    end_message = user_message(
        INVALID_PROMPT_INSTANCE_END_MESSAGE.format(instance_id=instance_id),
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(end_message)
    end_message_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=end_message_index,
            compressed_message_indices=[index for index, _ in active_items],
            active_context_indices=[],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-clear-context",
        )
    )
    return True


def message_swebench_instance_id(message: Any) -> str:
    details = getattr(message, "sidecar", {}).get("details", {})
    if not isinstance(details, Mapping):
        return ""
    swebench = details.get("swebench", {})
    if not isinstance(swebench, Mapping):
        return ""
    return str(swebench.get("instance_id") or "")
