"""Task delegation tool for sub-agent bundles.

The parent model sees one tool whose description lists every available
sub-agent (by `name` and `role`) and whose `subagent_type` enum parameter picks
one. The chosen sub-agent is run on the supplied `task` and its final message
text is returned as the tool result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from simple_agent_lab.messages import Message, message_text, system_message, text_of

from . import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result

if TYPE_CHECKING:
    from simple_agent_lab.core import Agent


def task_tool(
    agents: Sequence[Agent],
    *,
    name: str = "task",
    description: str | None = None,
    max_turns: int = 10,
    default_context: str | None = None,
) -> AgentTool:
    """Bundle several sub-agents into a single dispatch tool."""
    agent_list = list(agents)
    if not agent_list:
        raise ValueError("task_tool requires at least one sub-agent")

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

        def inject_context(messages: list[Message]) -> list[Message]:
            return [*context_messages, *messages]

        state, events = agent.run(
            task,
            max_turns=max_turns,
            transform=inject_context,
            abort=abort_flag,
        )
        for _ in events:
            if abort_flag():
                break
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
            )
        return text_result(text_of(final.content) or message_text(final))

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
            system_message(
                default_context.strip(),
                sender=tool_name,
                target=target,
                kind="context",
            )
        )
    if call_context:
        messages.append(
            system_message(
                call_context,
                sender=tool_name,
                target=target,
                kind="context",
            )
        )
    return messages
