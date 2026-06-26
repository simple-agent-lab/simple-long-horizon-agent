"""Codex-style ``spawn_agents`` parallel delegation tool.

Where the Claude Code-flavoured ``task`` tool (see ``task.py``) delegates to
*one* sub-agent per call, Codex's ``spawn_agents`` fans out to *several* at once:
the model hands over a batch of ``{ subagent_type, prompt }`` specs in a single
call and gets every sub-agent's final message back together. That batch shape is
the whole point of the comparison — it lets the parent decompose a problem and
explore branches in parallel instead of serializing one delegation at a time.

This coexists with ``task.py`` rather than replacing it, so the lab can run the
same sub-agents under both harness styles and compare.

Each spec runs the chosen sub-agent on its prompt and the final message rides
back as one labelled block in the combined result. Sub-agents are independent —
one failing (no final message produced) is reported inline and does not abort
the rest of the batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from simple_agent_lab.messages import message_text, text_of

from . import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result

if TYPE_CHECKING:
    from simple_agent_lab.core import Agent


SPAWN_AGENTS_TOOL_NAME = "spawn_agents"


def spawn_agents_tool(
    agents: Sequence[Agent],
    *,
    name: str = SPAWN_AGENTS_TOOL_NAME,
    description: str | None = None,
    max_turns: int = 10,
    max_agents: int = 8,
) -> AgentTool:
    """Bundle sub-agents into a Codex-style parallel fan-out tool.

    The returned tool accepts a batch of specs (capped at ``max_agents``) and
    runs each chosen sub-agent on its own prompt, returning every final message
    in one result.
    """

    agent_list = list(agents)
    if not agent_list:
        raise ValueError("spawn_agents_tool requires at least one sub-agent")
    if max_agents < 1:
        raise ValueError("max_agents must be >= 1")

    by_name: dict[str, Agent] = {}
    for agent in agent_list:
        if agent.name in by_name:
            raise ValueError(f"duplicate sub-agent name: {agent.name!r}")
        by_name[agent.name] = agent

    roster = "\n".join(
        f"- {agent.name}: {agent.role or '(no description)'}" for agent in agent_list
    )
    resolved_description = description or (
        "Spawn one or more sub-agents to work in parallel. Pass a batch of "
        "`agents`, each a self-contained `{subagent_type, prompt}` task; every "
        "sub-agent runs independently and its final message is returned. Put "
        f"everything a sub-agent needs into its `prompt` (max {max_agents} per "
        f"call).\nAvailable sub-agents:\n{roster}"
    )

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort_flag: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update

        specs = args.get("agents")
        if not isinstance(specs, list) or not specs:
            return text_result(
                "`agents` is required and must be a non-empty list of "
                "{subagent_type, prompt} specs.",
                is_error=True,
            )
        if len(specs) > max_agents:
            return text_result(
                f"Too many agents: {len(specs)} requested, limit is {max_agents}.",
                is_error=True,
            )

        parsed: list[tuple[str, str]] = []
        for position, spec in enumerate(specs):
            if not isinstance(spec, dict):
                return text_result(
                    f"agents[{position}] must be an object with subagent_type "
                    "and prompt.",
                    is_error=True,
                )
            subagent_type = str(spec.get("subagent_type", "")).strip()
            prompt = str(spec.get("prompt", "")).strip()
            if not prompt:
                return text_result(
                    f"agents[{position}].prompt is required and must be non-empty.",
                    is_error=True,
                )
            if subagent_type not in by_name:
                return text_result(
                    f"agents[{position}].subagent_type {subagent_type!r} is "
                    f"unknown. Available: {sorted(by_name)}",
                    is_error=True,
                )
            parsed.append((subagent_type, prompt))

        sections: list[str] = []
        sub_events: list[Any] = []
        any_error = False
        for position, (subagent_type, prompt) in enumerate(parsed):
            if abort_flag():
                sections.append(f"[agent {position} | {subagent_type}] aborted.")
                any_error = True
                break
            agent = by_name[subagent_type]
            state, events = agent.run(prompt, max_turns=max_turns, abort=abort_flag)
            for _ in events:
                if abort_flag():
                    break
            sub_events.append(
                {"subagent_type": subagent_type, "events": list(state.events)}
            )
            final = next(
                (m for m in reversed(state.messages) if m.kind == "final"),
                None,
            )
            header = f"[agent {position} | {subagent_type}]"
            if final is None:
                sections.append(f"{header} produced no final message.")
                any_error = True
            else:
                body = text_of(final.content) or message_text(final)
                sections.append(f"{header}\n{body}")

        return text_result(
            "\n\n".join(sections),
            is_error=any_error,
            details={"sub_events": sub_events, "count": len(parsed)},
        )

    return AgentTool(
        name=name,
        description=resolved_description,
        parameters={
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "description": "Batch of sub-agent tasks to run in parallel.",
                    "minItems": 1,
                    "maxItems": max_agents,
                    "items": {
                        "type": "object",
                        "properties": {
                            "subagent_type": {
                                "type": "string",
                                "enum": sorted(by_name),
                                "description": "Name of the sub-agent to invoke.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "Self-contained instruction for this "
                                    "sub-agent, including any context it needs."
                                ),
                            },
                        },
                        "required": ["subagent_type", "prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["agents"],
            "additionalProperties": False,
        },
        execute=execute,
    )
