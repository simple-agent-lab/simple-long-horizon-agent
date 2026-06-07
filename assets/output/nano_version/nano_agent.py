"""Nano agent runtime.

This file keeps only the idea:

    messages -> agent.generate -> optional tool calls -> tool_result messages

It is intentionally tiny. The real project has typed content blocks, provider
adapters, tracing, context compression, retries, and multimodal results. This
nano version keeps the boundary shape visible.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal


Kind = Literal["task", "step", "final", "tool_result"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    sender: str
    target: str
    kind: Kind
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    text: str
    is_error: bool = False


ToolFn = Callable[[dict[str, Any]], str]
GenerateFn = Callable[[list[Message]], Message]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    execute: ToolFn
    sequential: bool = False


@dataclass
class State:
    messages: list[Message] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, message: Message) -> None:
        self.messages.append(message)

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"event": name, **data})


@dataclass
class Agent:
    name: str
    generate: GenerateFn
    tools: tuple[AgentTool, ...] = ()
    system_prompt: str = ""


def run(agent: Agent, task: str, *, max_turns: int = 8) -> State:
    """Drive one agent until it emits a final message or runs out of turns."""

    state = State()
    state.record(Message(sender="user", target=agent.name, kind="task", text=task))
    tool_by_name = {tool.name: tool for tool in agent.tools}

    state.event("agent_start", agent=agent.name)
    for turn in range(max_turns):
        state.event("turn_start", agent=agent.name, turn=turn)

        # In the real framework this is build_context_view(...): compression,
        # headers, routing filters, and token accounting can happen here.
        visible = list(state.messages)
        state.event(
            "model_request",
            visible_messages=len(visible),
            tools=list(tool_by_name),
        )

        output = agent.generate(visible)
        state.record(output)
        state.event(
            "model_response",
            kind=output.kind,
            tool_calls=len(output.tool_calls),
        )

        if output.tool_calls:
            results = dispatch_tools(output.tool_calls, tool_by_name)
            state.record(
                Message(
                    sender="runtime",
                    target=agent.name,
                    kind="tool_result",
                    text="\n".join(result.text for result in results),
                )
            )

        state.event("turn_end", agent=agent.name, turn=turn)
        if output.kind == "final":
            state.event("agent_end", reason="done")
            return state

    state.event("agent_end", reason="max_turns")
    return state


def dispatch_tools(
    tool_calls: Iterable[ToolCall],
    tool_by_name: dict[str, AgentTool],
    *,
    max_workers: int = 8,
) -> list[ToolResult]:
    """Run tool calls.

    This is the small "executor" idea inside the agent loop. If a tool declares
    itself sequential, we collapse to one worker; otherwise independent tool
    calls may run in parallel.
    """

    calls = list(tool_calls)
    sequential = any(
        tool_by_name.get(call.name) and tool_by_name[call.name].sequential
        for call in calls
    )
    workers = 1 if sequential else min(max_workers, len(calls) or 1)
    results: dict[str, ToolResult] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute_one, call, tool_by_name): call
            for call in calls
        }
        for future in as_completed(futures):
            call = futures[future]
            try:
                results[call.id] = future.result()
            except Exception as exc:
                results[call.id] = ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    text=f"{type(exc).__name__}: {exc}",
                    is_error=True,
                )

    # Deterministic order matters for traces and replay.
    return [results[call.id] for call in calls]


def _execute_one(call: ToolCall, tool_by_name: dict[str, AgentTool]) -> ToolResult:
    tool = tool_by_name.get(call.name)
    if tool is None:
        return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", True)
    text = tool.execute(call.arguments)
    return ToolResult(call.id, call.name, text)

