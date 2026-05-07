"""Event-sourced graph runtime core.

This version makes graph routing and agent loops observable: graph, node, edge,
model, tool, message, and stop decisions are all runtime events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional

from simple_agent_lab.messages import (
    Message,
    ModelMessage,
    message_text,
    message_tool_calls,
    system_message,
    to_model_messages,
    tool_result_message,
    user_message,
)
from simple_agent_lab.tools import (
    AgentTool,
    Tool,
    ToolResult,
    text_result,
    tool_result_text,
)


MetaMode = Literal["none", "header"]
RuntimeEventKind = Literal[
    "message",
    "model_request",
    "model_response",
    "stop",
    "tool_execution_start",
    "tool_execution_end",
    "graph_start",
    "graph_end",
    "node_start",
    "node_end",
    "edge_traversed",
]
StopReason = Literal["final", "max_steps", "tool_terminate"]
GraphStopReason = Literal["done", "max_node_runs"]


@dataclass(frozen=True)
class RuntimeEvent:
    index: int
    kind: RuntimeEventKind
    data: dict[str, Any]


EventObserver = Callable[[RuntimeEvent], None]


@dataclass(frozen=True)
class Agent:
    name: str
    instruction: str


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 4
    last_messages: int | None = None
    meta: MetaMode = "header"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.last_messages is not None and self.last_messages < 1:
            raise ValueError("last_messages must be >= 1 when set")
        if self.meta not in {"none", "header"}:
            raise ValueError(f"meta must be 'none' or 'header', got {self.meta!r}")


@dataclass
class RuntimeState:
    task: str
    messages: list[Message] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)
    observers: list[EventObserver] = field(default_factory=list, repr=False)

    def emit(self, kind: RuntimeEventKind, **data: Any) -> RuntimeEvent:
        event = RuntimeEvent(len(self.events), kind, data)
        self.events.append(event)
        for observer in list(self.observers):
            observer(event)
        return event

    def subscribe(self, observer: EventObserver) -> Callable[[], None]:
        """Register a synchronous observer for future events."""
        self.observers.append(observer)

        def unsubscribe() -> None:
            if observer in self.observers:
                self.observers.remove(observer)

        return unsubscribe

    def record(self, message: Message) -> Message:
        self.messages.append(message)
        self.emit("message", message=message)
        return message

    def user(self, target: str, content: str, *, kind: str = "task") -> Message:
        return self.record(
            user_message(content, sender="user", target=target, kind=kind)
        )


@dataclass(frozen=True)
class RunResult:
    state: RuntimeState
    steps: int
    stop_reason: StopReason


@dataclass(frozen=True)
class RunReport:
    stop_reason: StopReason | None
    steps: int
    model_calls: int
    tool_calls: int
    tool_errors: int
    messages: int
    max_visible_count: int
    output_kinds: tuple[str, ...]
    graph_stop_reason: GraphStopReason | None = None
    node_runs: int = 0
    graph_path: tuple[str, ...] = ()


class ModelClient:
    def generate(
        self,
        messages: list[Message],
        meta: MetaMode = "none",
        tools: Optional[list["Tool"]] = None,
    ) -> Message:
        raise NotImplementedError


EdgeCondition = Callable[[RuntimeState], bool]


@dataclass(frozen=True)
class GraphNode:
    name: str
    agent: Agent
    model: ModelClient
    tools: tuple[AgentTool, ...] = ()
    config: RunConfig | None = None


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str | None = None
    label: str = ""
    condition: EdgeCondition | None = None


@dataclass(frozen=True)
class AgentGraph:
    entry: str
    nodes: dict[str, GraphNode]
    edges: tuple[GraphEdge, ...] = ()
    max_node_runs: int = 12

    def __post_init__(self) -> None:
        if self.entry not in self.nodes:
            raise ValueError(f"entry node {self.entry!r} is not in graph")
        if self.max_node_runs < 1:
            raise ValueError("max_node_runs must be >= 1")
        for node_name, node in self.nodes.items():
            if node_name != node.name:
                raise ValueError(
                    f"node key {node_name!r} must match GraphNode.name {node.name!r}"
                )
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"edge source {edge.source!r} is not in graph")
            if edge.target is not None and edge.target not in self.nodes:
                raise ValueError(f"edge target {edge.target!r} is not in graph")

    def outgoing(self, source: str) -> tuple[GraphEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source == source)


@dataclass(frozen=True)
class GraphRunResult:
    state: RuntimeState
    node_runs: int
    stop_reason: GraphStopReason
    path: tuple[str, ...]


def make_tool_result_message(
    call_id: str,
    tool_name: str,
    result: ToolResult,
    *,
    target: str,
) -> Message:
    """Convert a `ToolResult` into a `kind="tool_result"` message for the log."""
    text = tool_result_text(result)
    return tool_result_message(
        text,
        tool_call_id=call_id,
        tool_name=tool_name,
        sender=tool_name,
        target=target,
        is_error=result.is_error,
        data={
            "details": result.details,
            "content_blocks": [
                {"kind": c.kind, "text": c.text, "image_url": c.image_url}
                for c in result.content
            ],
        },
    )


def dispatch_tool_calls(
    assistant_msg: Message,
    tools: dict[str, AgentTool],
    state: "RuntimeState",
) -> bool:
    """Run the explicit tool calls carried by an assistant message.

    Records a `kind="tool_result"` message per call. Exceptions from
    `execute` are caught and turned into `is_error=True` results — they
    never bubble out. Returns True if any tool requested termination.

    Sequential and synchronous; no parallel / streaming. (03 prizes the
    short, easy-to-read loop over the 02 dispatcher's threading.)
    """
    tool_calls = message_tool_calls(assistant_msg)
    if not tool_calls:
        return False

    target = assistant_msg.sender or "assistant"
    terminated = False

    for tc in tool_calls:
        call_id = tc.id
        name = tc.name
        args = dict(tc.arguments)

        state.emit("tool_execution_start", tool_call_id=call_id, tool_name=name)

        tool = tools.get(name)
        if tool is None:
            result = text_result(f"Tool {name!r} not found", is_error=True)
        elif tool.execute is None:
            result = text_result(
                f"Tool {name!r} has no execute function",
                is_error=True,
            )
        else:
            try:
                result = tool.execute(call_id, args)
            except Exception as exc:
                result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)

        state.emit(
            "tool_execution_end",
            tool_call_id=call_id,
            tool_name=name,
            is_error=result.is_error,
            terminate=result.terminate,
        )
        state.record(make_tool_result_message(call_id, name, result, target=target))
        if result.terminate:
            terminated = True

    return terminated


def context_view(agent: Agent, state: RuntimeState, config: RunConfig) -> list[Message]:
    visible = [
        message
        for message in state.messages
        if message.target in {agent.name, "all"} or message.sender == agent.name
    ]
    if config.last_messages is not None:
        visible = visible[-config.last_messages :]
    return [
        system_message(
            agent.instruction,
            sender="system",
            target=agent.name,
            kind="instruction",
        ),
        *visible,
    ]


def model_messages(messages: list[Message], meta: MetaMode = "none") -> list[ModelMessage]:
    if meta not in {"none", "header"}:
        raise ValueError(f"meta must be 'none' or 'header', got {meta!r}")

    return to_model_messages(messages, with_header=meta == "header")


def _tool_specs(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def as_agent_message(agent: Agent, output: Message) -> Message:
    return replace(
        output,
        sender=agent.name,
        target="user" if output.kind == "final" else agent.name,
        data={**dict(output.data), "model_sender": output.sender},
    )


def messages_from_events(events: list[RuntimeEvent]) -> list[Message]:
    """Replay transcript messages from an event log."""
    messages: list[Message] = []
    for event in events:
        if event.kind != "message":
            continue
        message = event.data.get("message")
        if message is not None:
            messages.append(message)
    return messages


def run_report(state: RuntimeState) -> RunReport:
    """Summarize one run using only `RuntimeState.events`."""
    requests = [event for event in state.events if event.kind == "model_request"]
    responses = [event for event in state.events if event.kind == "model_response"]
    tool_ends = [event for event in state.events if event.kind == "tool_execution_end"]
    stop = next((event for event in reversed(state.events) if event.kind == "stop"), None)
    graph_end = next(
        (event for event in reversed(state.events) if event.kind == "graph_end"),
        None,
    )
    return RunReport(
        stop_reason=stop.data["reason"] if stop is not None else None,
        steps=int(stop.data["steps"]) if stop is not None else 0,
        model_calls=len(requests),
        tool_calls=len(tool_ends),
        tool_errors=sum(1 for event in tool_ends if event.data.get("is_error")),
        messages=len(messages_from_events(state.events)),
        max_visible_count=max(
            (int(event.data.get("visible_count", 0)) for event in requests),
            default=0,
        ),
        output_kinds=tuple(
            str(event.data.get("output_kind", "")) for event in responses
        ),
        graph_stop_reason=(
            graph_end.data.get("reason") if graph_end is not None else None
        ),
        node_runs=(
            int(graph_end.data.get("node_runs", 0)) if graph_end is not None else 0
        ),
        graph_path=tuple(graph_end.data.get("path", ())) if graph_end is not None else (),
    )


class AgentLoop:
    def __init__(self, config: RunConfig = RunConfig()):
        self.config = config

    def run(
        self,
        agent: Agent,
        state: RuntimeState,
        model: ModelClient,
        tools: Optional[list[AgentTool]] = None,
    ) -> RunResult:
        tool_registry: dict[str, AgentTool] = {t.name: t for t in (tools or [])}
        wire_tools: list[Tool] = list(tools) if tools else []

        for step in range(1, self.config.max_steps + 1):
            visible = context_view(agent, state, self.config)
            model_payload = model_messages(visible, meta=self.config.meta)
            state.emit(
                "model_request",
                agent=agent.name,
                visible_count=len(visible),
                model_message_count=len(model_payload),
                tool_count=len(wire_tools),
                tools=_tool_specs(wire_tools),
                model_payload=model_payload,
            )
            output = model.generate(
                visible,
                meta=self.config.meta,
                tools=wire_tools or None,
            )
            state.emit(
                "model_response",
                model_sender=output.sender,
                output_kind=output.kind,
                tool_call_count=len(message_tool_calls(output)),
            )
            message = as_agent_message(agent, output)
            state.record(message)

            # Dispatch any tool_calls embedded in the assistant message.
            if tool_registry and message_tool_calls(message):
                terminated = dispatch_tool_calls(message, tool_registry, state)
                if terminated:
                    state.emit("stop", reason="tool_terminate", steps=step)
                    return RunResult(state, step, "tool_terminate")
                # Tool calls do not stop the loop — model gets results next turn.
                continue

            if message.kind == "final":
                state.emit("stop", reason="final", steps=step)
                return RunResult(state, step, "final")
        state.emit("stop", reason="max_steps", steps=self.config.max_steps)
        return RunResult(state, self.config.max_steps, "max_steps")


class GraphRunner:
    """Run an AgentGraph by delegating each node to the normal AgentLoop."""

    def __init__(self, default_config: RunConfig = RunConfig()):
        self.default_config = default_config

    def run(self, graph: AgentGraph, state: RuntimeState) -> GraphRunResult:
        current = graph.entry
        path: list[str] = []
        state.emit(
            "graph_start",
            entry=graph.entry,
            nodes=tuple(graph.nodes),
            max_node_runs=graph.max_node_runs,
        )

        for node_run in range(1, graph.max_node_runs + 1):
            node = graph.nodes[current]
            path.append(current)
            state.emit(
                "node_start",
                node=current,
                agent=node.agent.name,
                tools=tuple(tool.name for tool in node.tools),
            )
            node_result = AgentLoop(node.config or self.default_config).run(
                node.agent,
                state,
                node.model,
                tools=list(node.tools),
            )
            state.emit(
                "node_end",
                node=current,
                agent=node.agent.name,
                stop_reason=node_result.stop_reason,
                steps=node_result.steps,
            )

            edge = self._next_edge(graph, current, state)
            if edge is None:
                state.emit(
                    "graph_end",
                    reason="done",
                    node_runs=node_run,
                    path=tuple(path),
                )
                return GraphRunResult(state, node_run, "done", tuple(path))
            if edge.target is None:
                state.emit(
                    "edge_traversed",
                    source=edge.source,
                    target=None,
                    label=edge.label,
                )
                state.emit(
                    "graph_end",
                    reason="done",
                    node_runs=node_run,
                    path=tuple(path),
                )
                return GraphRunResult(state, node_run, "done", tuple(path))

            state.emit(
                "edge_traversed",
                source=edge.source,
                target=edge.target,
                label=edge.label,
            )
            self._record_handoff(state, source=current, target=edge.target)
            current = edge.target

        state.emit(
            "graph_end",
            reason="max_node_runs",
            node_runs=graph.max_node_runs,
            path=tuple(path),
        )
        return GraphRunResult(state, graph.max_node_runs, "max_node_runs", tuple(path))

    def _next_edge(
        self,
        graph: AgentGraph,
        source: str,
        state: RuntimeState,
    ) -> GraphEdge | None:
        for edge in graph.outgoing(source):
            if edge.condition is None or edge.condition(state):
                return edge
        return None

    def _record_handoff(self, state: RuntimeState, *, source: str, target: str) -> None:
        if not state.messages:
            return
        message = state.messages[-1]
        state.record(
            user_message(
                f"Handoff from {source}: {message_text(message)}",
                sender="graph",
                target=target,
                kind="handoff",
                channel="graph",
            )
        )


def print_trace(state: RuntimeState) -> None:
    for event in state.events:
        if event.kind == "graph_start":
            print(
                f"{event.index:02d} graph_start   "
                f"entry={event.data.get('entry')} "
                f"nodes={list(event.data.get('nodes', ()))}"
            )
        elif event.kind == "graph_end":
            print(
                f"{event.index:02d} graph_end     "
                f"reason={event.data.get('reason')} "
                f"path={list(event.data.get('path', ()))} "
                f"node_runs={event.data.get('node_runs')}"
            )
        elif event.kind == "node_start":
            print(
                f"{event.index:02d} node_start    "
                f"node={event.data.get('node')} "
                f"agent={event.data.get('agent')} "
                f"tools={list(event.data.get('tools', ()))}"
            )
        elif event.kind == "node_end":
            print(
                f"{event.index:02d} node_end      "
                f"node={event.data.get('node')} "
                f"reason={event.data.get('stop_reason')} "
                f"steps={event.data.get('steps')}"
            )
        elif event.kind == "edge_traversed":
            print(
                f"{event.index:02d} edge          "
                f"{event.data.get('source')} -> {event.data.get('target')} "
                f"label={event.data.get('label')}"
            )
        elif event.kind == "message":
            message = event.data["message"]
            print(
                f"{event.index:02d} message        "
                f"{message.kind:<8} {message.sender:>10} {message_text(message)}"
            )
        elif event.kind == "model_request":
            print(
                f"{event.index:02d} model_request  "
                f"agent={event.data.get('agent')} "
                f"visible={event.data.get('visible_count')} "
                f"model_messages={event.data.get('model_message_count')} "
                f"tools={event.data.get('tool_count')}"
            )
        elif event.kind == "model_response":
            print(
                f"{event.index:02d} model_response "
                f"model={event.data.get('model_sender')} "
                f"kind={event.data.get('output_kind')} "
                f"tool_calls={event.data.get('tool_call_count')}"
            )
        elif event.kind == "tool_execution_start":
            print(
                f"{event.index:02d} tool_start     "
                f"id={event.data.get('tool_call_id')} "
                f"name={event.data.get('tool_name')}"
            )
        elif event.kind == "tool_execution_end":
            print(
                f"{event.index:02d} tool_end       "
                f"id={event.data.get('tool_call_id')} "
                f"name={event.data.get('tool_name')} "
                f"is_error={event.data.get('is_error')} "
                f"terminate={event.data.get('terminate')}"
            )
        elif event.kind == "stop":
            print(
                f"{event.index:02d} stop           "
                f"reason={event.data.get('reason')} "
                f"steps={event.data.get('steps')}"
            )
        else:
            print(f"{event.index:02d} {event.kind:<14} {event.data}")


def print_report(report: RunReport) -> None:
    print("\nreport")
    print("------")
    fields = [
        f"stop={report.stop_reason}",
        f"steps={report.steps}",
        f"model_calls={report.model_calls}",
        f"tool_calls={report.tool_calls}",
        f"tool_errors={report.tool_errors}",
        f"messages={report.messages}",
        f"max_visible={report.max_visible_count}",
        f"output_kinds={list(report.output_kinds)}",
    ]
    if report.graph_path:
        fields.append(f"graph_stop={report.graph_stop_reason}")
        fields.append(f"node_runs={report.node_runs}")
        fields.append(f"graph_path={' -> '.join(report.graph_path)}")
    print(
        " ".join(fields)
    )
