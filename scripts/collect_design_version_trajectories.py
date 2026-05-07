"""Collect raw trajectories for the temporary 01 / 02 / 03 design versions.

Run from the repo root:

    PYTHONPATH=src python3 scripts/collect_design_version_trajectories.py

This script is intentionally the runtime-specific adapter layer. The core
trajectory schema in `simple_agent_lab.trajectory` does not know about these
three runtime sketches.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import argparse
import importlib.util
import sys
from types import ModuleType
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.messages import Message
from simple_agent_lab.tools import AgentTool
from simple_agent_lab.trajectory import (
    ModelTurn,
    RunTrace,
    json_safe,
    trace_record,
    write_jsonl,
)


_MISSING = object()


@contextmanager
def loaded_demo(module_name: str, path: Path) -> Iterator[ModuleType]:
    """Load one design-version demo with its local `core.py` imports isolated."""
    saved_path = list(sys.path)
    module_keys = {module_name, "core", "agent", "models"}
    saved_modules = {key: sys.modules.get(key, _MISSING) for key in module_keys}
    for key in module_keys:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.path = saved_path
        for key, value in saved_modules.items():
            if value is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def collect_01() -> RunTrace:
    trace_id = "01_functional_loop.shopping_helper"
    producer = "design_version:01_functional_loop"
    path = ROOT / "examples/design_versions/01_functional_loop/demo.py"
    with loaded_demo("design_01_demo", path) as demo:
        calls: list[tuple[list[Message], list[AgentTool], Message]] = []
        base_model = demo.shopping_helper_model()

        def model(messages: list[Message], tools: list[AgentTool]) -> Message:
            output = base_model(messages, tools)
            calls.append((list(messages), list(tools), output))
            return output

        messages = demo.run_demo(model=model)

    model_turns = [
        ModelTurn(
            step_id=f"{trace_id}.model{index}",
            agent="assistant",
            input_messages=json_safe(input_messages),
            output_message=json_safe(output),
            tools=json_safe([_tool_spec(tool) for tool in tools]),
            meta={"model_call_index": index},
        )
        for index, (input_messages, tools, output) in enumerate(calls, start=1)
    ]
    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=demo.TASK,
        messages=json_safe(messages),
        events=[],
        model_turns=model_turns,
        meta={"source": "message-list"},
    )


def collect_02() -> RunTrace:
    trace_id = "02_balanced_runtime.agent_as_tool"
    producer = "design_version:02_balanced_runtime"
    path = ROOT / "examples/design_versions/02_balanced_runtime/demo.py"
    with loaded_demo("design_02_demo", path) as demo:
        runtime_obj = demo.run_demo()
        state = runtime_obj.state

    model_turns = _model_turns_from_events(trace_id=trace_id, events=state.events)
    for index, message in enumerate(state.messages, start=1):
        if getattr(message, "role", "") != "tool_result":
            continue
        details = message.data.get("details") or {}
        child_events = details.get("child_events") or []
        if not child_events:
            continue
        agent_name = str(details.get("agent_name", "child_agent"))
        model_turns.extend(_model_turns_from_events(
            trace_id=trace_id,
            events=child_events,
            prefix=f"child.{agent_name}.",
            extra_meta={"parent_message_index": index},
        ))

    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=state.task,
        messages=json_safe(state.messages),
        events=json_safe(state.events),
        model_turns=model_turns,
        meta={"source": "state.events"},
    )


def collect_03() -> RunTrace:
    trace_id = "03_event_runtime.weather_graph"
    producer = "design_version:03_event_runtime"
    path = ROOT / "examples/design_versions/03_event_runtime/demo.py"
    with loaded_demo("design_03_demo", path) as demo:
        result, observed_events = demo.run_demo()
        state = result.state

    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=state.task,
        messages=json_safe(state.messages),
        events=json_safe(state.events),
        model_turns=_model_turns_from_events(trace_id=trace_id, events=state.events),
        meta={"source": "state.events", "observed_events": observed_events},
    )


def _model_turns_from_events(
    *,
    trace_id: str,
    events: list[Any],
    prefix: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> list[ModelTurn]:
    turns: list[ModelTurn] = []
    pending: dict[str, Any] | None = None
    model_call_index = 0

    for event in events:
        kind = getattr(event, "kind", "")
        data = _event_data(event)
        if kind == "model_request":
            model_call_index += 1
            pending = {
                "agent": str(data.get("agent", "")),
                "input_messages": (
                    data.get("llm_payload")
                    or data.get("model_payload")
                    or data.get("visible")
                    or []
                ),
                "tools": data.get("tools") or [],
                "request_event_index": getattr(event, "index", None),
                "meta": {
                    "visible_count": data.get("visible_count"),
                    "model_message_count": (
                        data.get("llm_message_count")
                        or data.get("model_message_count")
                    ),
                    "tool_count": data.get("tool_count"),
                    "candidate_id": data.get("candidate_id"),
                },
            }
            continue

        if kind != "message" or pending is None:
            continue

        message = _event_message(event)
        if message is None or getattr(message, "role", "") != "assistant":
            continue
        agent = pending["agent"] or str(getattr(message, "sender", ""))
        if agent and getattr(message, "sender", "") != agent:
            continue

        turns.append(ModelTurn(
            step_id=f"{trace_id}.{prefix}model{model_call_index}",
            agent=agent,
            input_messages=json_safe(pending["input_messages"]),
            output_message=json_safe(message),
            tools=json_safe(pending["tools"]),
            meta={
                **pending["meta"],
                **dict(extra_meta or {}),
                "request_event_index": pending["request_event_index"],
                "message_event_index": getattr(event, "index", None),
            },
        ))
        pending = None

    return turns


def _event_data(event: Any) -> dict[str, Any]:
    if hasattr(event, "payload"):
        return dict(getattr(event, "payload"))
    return dict(getattr(event, "data", {}))


def _event_message(event: Any) -> Message | None:
    if hasattr(event, "message"):
        message = getattr(event, "message")
        return message if message is not None else None
    data = getattr(event, "data", {})
    return data.get("message")


def _tool_spec(tool: AgentTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def collect_all() -> list[RunTrace]:
    return [collect_01(), collect_02(), collect_03()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default=str(ROOT / "evals/out/design_version_trajectories.jsonl"),
        help="Where to write trajectory JSONL records.",
    )
    args = parser.parse_args()

    traces = collect_all()
    write_jsonl(args.jsonl, [trace_record(trace) for trace in traces])

    print(f"wrote {len(traces)} trajectories to {args.jsonl}")
    for trace in traces:
        print(
            f"{trace.trace_id}: producer={trace.producer} "
            f"model_turns={len(trace.model_turns)}"
        )


if __name__ == "__main__":
    main()
