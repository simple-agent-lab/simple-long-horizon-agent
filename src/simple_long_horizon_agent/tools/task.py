"""Task delegation tool for sub-agent bundles.

The parent model sees one tool whose description lists every available
sub-agent (by `name` and `role`) and whose `subagent_type` enum parameter picks
one. The chosen sub-agent is run on the supplied `task` and its final message
text is returned as the tool result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from simple_long_horizon_agent.messages import (
    Message,
    message_text,
    runtime_message,
    text_of,
)
from simple_long_horizon_agent.protocols import TurnEndEvent

from . import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result

if TYPE_CHECKING:
    from simple_long_horizon_agent.core import Agent


def task_tool(
    agents: Sequence[Agent],
    *,
    name: str = "task",
    description: str | None = None,
    max_turns: int = 10,
    soft_turn_limit: int | None = None,
    default_context: str | None = None,
) -> AgentTool:
    """Bundle several sub-agents into a single dispatch tool.

    When ``soft_turn_limit`` is set below ``max_turns``, the sub-agent receives
    one model-visible reminder after that many completed turns. The reminder
    does not add a model turn: it is appended between turns, and the existing
    ``max_turns`` value remains the hard stop.
    """
    agent_list = list(agents)
    if not agent_list:
        raise ValueError("task_tool requires at least one sub-agent")
    if soft_turn_limit is not None and soft_turn_limit < 1:
        raise ValueError("soft_turn_limit must be at least 1 or None")

    by_name: dict[str, Agent] = {}
    for agent in agent_list:
        if agent.name in by_name:
            raise ValueError(f"duplicate sub-agent name: {agent.name!r}")
        by_name[agent.name] = agent

    listing = "\n".join(
        f"- {agent.name}: {agent.role or '(no description)'}" for agent in agent_list
    )
    context_note = (
        "\nConfigured default context is passed to the selected sub-agent."
        if default_context and default_context.strip()
        else ""
    )
    resolved_description = description or (
        "Delegate a task to one of the available sub-agents.\n"
        f"Available sub-agents:\n{listing}"
        f"{context_note}"
    )

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort_flag: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        subagent_type = str(args.get("subagent_type", "")).strip()
        task = str(args.get("task", "")).strip()
        call_context = str(args.get("context", "")).strip()
        if not task:
            return text_result(
                "`task` is required and must be non-empty",
                is_error=True,
            )
        agent = by_name.get(subagent_type)
        if agent is None:
            return text_result(
                f"Unknown subagent_type {subagent_type!r}. "
                f"Available: {sorted(by_name)}",
                is_error=True,
            )

        context_messages = _context_messages(
            tool_name=name,
            target=agent.name,
            default_context=default_context,
            call_context=call_context,
        )

        state, events = agent.run(
            task,
            max_turns=max_turns,
            abort=abort_flag,
        )
        # Record the sub-agent context onto `state` before driving `events`.
        # Recording up-front (rather than a per-turn transform) keeps every
        # message the sub-agent actually sees visible in `state.messages`
        # and in the trace -- no hidden in-flight injection.
        for context_message in context_messages:
            state.record(context_message)
        completed_turns = 0
        for event in events:
            if abort_flag():
                break
            if isinstance(event, TurnEndEvent):
                completed_turns += 1
                should_remind = (
                    soft_turn_limit is not None
                    and completed_turns == soft_turn_limit
                    and completed_turns < max_turns
                    and not event.terminated
                    and not any(message.kind == "final" for message in state.messages)
                )
                if should_remind:
                    state.record(
                        _turn_limit_reminder(
                            tool_name=name,
                            target=agent.name,
                            turns_remaining=max_turns - completed_turns,
                        )
                    )
        final = next(
            (
                message
                for message in reversed(state.messages)
                if message.kind == "final"
            ),
            None,
        )
        if final is None:
            return text_result(
                f"Sub-agent {agent.name!r} produced no final message",
                is_error=True,
                details={"sub_events": list(state.events)},
            )
        return text_result(
            text_of(final.content) or message_text(final),
            details={"sub_events": list(state.events)},
        )

    return AgentTool(
        name=name,
        description=resolved_description,
        parameters={
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "enum": sorted(by_name),
                    "description": "Name of the sub-agent to invoke.",
                },
                "task": {
                    "type": "string",
                    "description": "Task to delegate to the chosen sub-agent.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional extra context to pass to the chosen sub-agent "
                        "for this delegated task."
                    ),
                },
            },
            "required": ["subagent_type", "task"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _context_messages(
    *,
    tool_name: str,
    target: str,
    default_context: str | None,
    call_context: str,
) -> list[Message]:
    messages: list[Message] = []
    if default_context and default_context.strip():
        messages.append(
            runtime_message(
                default_context.strip(),
                sender=tool_name,
                target=target,
                kind="context",
            )
        )
    if call_context:
        messages.append(
            runtime_message(
                call_context,
                sender=tool_name,
                target=target,
                kind="context",
            )
        )
    return messages


def _turn_limit_reminder(
    *, tool_name: str, target: str, turns_remaining: int
) -> Message:
    return runtime_message(
        f"You have {turns_remaining} turns remaining before this delegated "
        "task reaches its hard limit. Stop broad exploration, focus on "
        "completing the requested work, run only essential checks, and return "
        "a concise final summary before the limit.",
        sender=tool_name,
        target=target,
        kind="context",
    )
