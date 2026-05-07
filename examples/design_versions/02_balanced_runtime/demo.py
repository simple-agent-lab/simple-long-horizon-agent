"""Balanced runtime demo — agent-as-tool delegation.

A coordinator agent can call a `run_agent` tool. The tool launches a focused
child agent, waits for its result, and returns that result as a normal
`tool_result` message.

This follows the Claude Code AgentTool shape:

    coordinator step
      -> tool_call: run_agent(agent_name="tweet_writer", prompt=...)
      -> tool execution runs tweet_writer as a child agent
      -> tool_result returns to coordinator
      -> coordinator finalizes

Swap `Provider(api="fake", ...)` for a real provider once an adapter is
registered (Anthropic / OpenAI). Nothing else in this file changes.
"""

from __future__ import annotations

from typing import Any, Callable

from agent import AgentRuntime
from core import (
    Agent,
    Event,
    State,
    last_message,
    make_llm_step,
    print_trace,
    run_to_completion,
    sequence,
)
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    text_result,
)


_FAKE = LLMProvider(id="fake", api="fake", model="fake-model")
TASK = "Write a tweet about Python decorators."


def tweet_writer_agent() -> Agent:
    return Agent(
        name="tweet_writer",
        role="Write one concise tweet.",
        step=make_llm_step(
            _FAKE,
            system_prompt="You polish a draft into a single tweet (<=280 chars).",
            target="user",
        ),
    )


def run_agent_tool() -> AgentTool:
    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Subagent run aborted before start.", is_error=True)

        agent_name = str(args.get("agent_name", "tweet_writer"))
        if agent_name != "tweet_writer":
            return text_result(f"Unknown subagent {agent_name!r}.", is_error=True)

        prompt = str(args.get("prompt", "")).strip()
        child = tweet_writer_agent()
        child_state = State(prompt)
        child_state.send("task", "user", child.name, prompt)
        run_to_completion(
            {child.name: child},
            child_state,
            sequence(child.name),
        )
        result = last_message(child_state, sender=child.name)
        return text_result(
            str(result.content),
            details={
                "agent_name": child.name,
                "child_event_count": len(child_state.events),
                "child_events": child_state.events,
            },
        )

    return AgentTool(
        name="run_agent",
        description="Run a focused child agent and return its final result.",
        parameters={
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Child agent to run, e.g. tweet_writer.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Task prompt for the child agent.",
                },
            },
            "required": ["agent_name", "prompt"],
        },
        execute=execute,
        label="Run child agent",
        execution_mode="sequential",
    )


def coordinator_agent() -> Agent:
    return Agent(
        name="coordinator",
        role="Delegate focused writing work to child agents.",
        step=make_llm_step(
            _FAKE,
            system_prompt=(
                "You are a coordinator. Use `run_agent` for focused writing "
                "tasks, then return the child agent result to the user."
            ),
            target="user",
        ),
    )


def coordinator_until_final(state: State) -> str | None:
    if any(
        message.sender == "coordinator" and message.kind == "final"
        for message in state.messages
    ):
        return None

    turns = sum(
        1
        for event in state.events
        if event.kind == "turn_end" and event.data.get("agent") == "coordinator"
    )
    return "coordinator" if turns < 3 else None


def run_demo(on_event: Callable[[Event], None] | None = None) -> AgentRuntime:
    runtime = AgentRuntime([coordinator_agent()], tools=[run_agent_tool()])
    stream = runtime.prompt(
        TASK,
        target="coordinator",
        next_agent=coordinator_until_final,
    )
    for event in stream:
        if on_event is not None:
            on_event(event)
    return runtime


def main() -> None:
    def print_live_event(event: Event) -> None:
        if event.kind == "message":
            message = event.message
            if message is not None and message.sender in {"coordinator", "run_agent"}:
                print(f"  [{message.sender:>11}] {message.content}")
        elif event.kind == "tool_execution_start":
            print(f"  [       tool] start {event.data.get('tool_name')}")
        elif event.kind == "tool_execution_end":
            print(f"  [       tool] end {event.data.get('tool_name')}")

    print("=== agent-as-tool delegation ===")
    runtime = run_demo(on_event=print_live_event)

    print("\n=== full trace ===")
    print_trace(runtime.state)


if __name__ == "__main__":
    main()
