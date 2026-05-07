"""Event-runtime demo — a tiny graph of cooperating agents.

Realistic shape a user would write:

  1. Define a `get_weather` tool.
  2. Build an `LLMModelClient` against the LLM layer.
  3. Put two agents into an `AgentGraph`.
  4. Run the graph against a `RuntimeState` seeded with the user's question.

Demonstrates the event runtime's distinctive features:
  * `state.events` records every model_request / model_response /
    message / tool_execution_* / graph / node / edge / stop event
  * `state.subscribe(...)` observes events as they are emitted
  * `run_report(...)` summarizes model/tool/message counts from events
  * `messages_from_events(...)` replays the transcript from the event log
  * `GraphRunner` composes normal `AgentLoop` runs into a small graph
  * tool dispatch is automatic when the assistant message contains
    tool_calls; the loop continues so the model can react to results
"""

from __future__ import annotations

from typing import Any

from core import (
    Agent,
    AgentGraph,
    GraphEdge,
    GraphNode,
    GraphRunner,
    MetaMode,
    ModelClient,
    RunConfig,
    RuntimeState,
    messages_from_events,
    print_report,
    print_trace,
    run_report,
)
from models import LLMModelClient
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.messages import Message, assistant_message, message_text
from simple_agent_lab.tools import AgentTool, ToolContent, ToolResult


# --------------------------------------------------------------------------
# Tool: a canned weather lookup. A real tool would call a network service.

_FAKE_WEATHER = {
    "Tokyo": "18°C, partly cloudy, 60% humidity",
    "New York": "8°C, light rain, 70% humidity",
    "London": "12°C, overcast, 80% humidity",
}
TASK = "Tell me what to wear in Tokyo today."


def get_weather(call_id: str, args: dict[str, Any]) -> ToolResult:
    city = args.get("city", "")
    forecast = _FAKE_WEATHER.get(city, f"No data available for {city!r}")
    return ToolResult(
        content=[ToolContent(text=forecast)],
        details={"city": city, "source": "demo-fake"},
        is_error=city not in _FAKE_WEATHER,
    )


weather_tool = AgentTool(
    name="get_weather",
    description="Look up current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Tokyo'.",
            },
        },
        "required": ["city"],
    },
    execute=get_weather,
)


# --------------------------------------------------------------------------
# Model: a normal LLM client. The fake adapter is deterministic, but it still
# reads the same messages, tool definitions, and tool results that a real
# provider adapter would see.


class HandoffAdvisorModel(ModelClient):
    """Tiny local adapter showing that graph nodes can use different clients."""

    def generate(
        self,
        messages: list[Message],
        meta: MetaMode = "none",
        tools: list[Any] | None = None,
    ) -> Message:
        del meta, tools
        handoff = next(
            (
                message_text(message)
                for message in reversed(messages)
                if message.kind == "handoff"
            ),
            "",
        )
        advice_seed = (
            handoff.split(":", 1)[-1].strip() or "The weather looks mild."
        )
        return assistant_message(
            (
                f"{advice_seed} Wear breathable layers, carry a light jacket, "
                "and choose comfortable walking shoes."
            ),
            sender="local-advisor-model",
            target="travel_advisor",
            kind="final",
            data={"provider": "local-rule"},
        )


def has_weather_result(state: RuntimeState) -> bool:
    return any(
        message.kind == "tool_result"
        and getattr(message, "tool_name", "") == "get_weather"
        and not getattr(message, "is_error", False)
        for message in state.messages
    )


def run_demo() -> tuple[Any, list[str]]:
    provider = LLMProvider(id="fake", api="fake", model="fake-model")
    researcher_instruction = (
        "You are a weather researcher. Use `get_weather` before answering."
    )
    advisor_instruction = (
        "You are a concise travel advisor. Turn the handoff into clothing advice."
    )

    researcher_model = LLMModelClient(provider, system_prompt=researcher_instruction)
    advisor_model = HandoffAdvisorModel()
    researcher = Agent("weather_researcher", researcher_instruction)
    advisor = Agent("travel_advisor", advisor_instruction)
    state = RuntimeState(TASK)
    observed_events: list[str] = []
    state.subscribe(lambda event: observed_events.append(event.kind))
    state.user(researcher.name, state.task)

    graph = AgentGraph(
        entry=researcher.name,
        nodes={
            researcher.name: GraphNode(
                name=researcher.name,
                agent=researcher,
                model=researcher_model,
                tools=(weather_tool,),
            ),
            advisor.name: GraphNode(
                name=advisor.name,
                agent=advisor,
                model=advisor_model,
            ),
        },
        edges=(
            GraphEdge(
                source=researcher.name,
                target=advisor.name,
                label="weather found",
                condition=has_weather_result,
            ),
            GraphEdge(
                source=researcher.name,
                target=None,
                label="no weather result",
            ),
        ),
        max_node_runs=4,
    )

    result = GraphRunner(RunConfig(max_steps=4, meta="header")).run(graph, state)
    return result, observed_events


def main() -> None:
    result, observed_events = run_demo()
    print_trace(result.state)
    print_report(run_report(result.state))
    replayed = messages_from_events(result.state.events)
    print(
        f"replayed_messages={len(replayed)} "
        f"observed_events={len(observed_events)}"
    )
    print(f"graph_path={' -> '.join(result.path)}")
    print(f"\nstop_reason={result.stop_reason}, node_runs={result.node_runs}")


if __name__ == "__main__":
    main()
