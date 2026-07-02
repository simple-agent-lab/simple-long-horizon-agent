"""Trace helpers for multi-agent workflow facades."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from simple_agent_lab.messages import TextBlock, ToolCallBlock, ToolResultBlock
from simple_agent_lab.protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.trace import (
    event_stream_bytes,
    run_trace_from_state,
)

from .base import WorkflowResult, state_output_tokens

ArtifactPut = Callable[[str, bytes], None]


def workflow_steps_breakdown(result: WorkflowResult, workflow: str) -> dict[str, Any]:
    """Return a compact per-step breakdown for a workflow run."""

    seen: set[int] = set()
    total = 0
    for step in result.steps:
        if id(step.state) not in seen:
            seen.add(id(step.state))
            total += state_output_tokens(step.state)
    return {
        "workflow": workflow,
        "output_tokens": total,
        "steps": [
            {
                "name": step.name,
                "role": step.role,
                "output_tokens": state_output_tokens(step.state),
                "output": step.output,
            }
            for step in result.steps
        ],
    }


def write_workflow_subagent_traces(
    result: WorkflowResult,
    workflow: str,
    put: ArtifactPut,
) -> list[dict[str, Any]]:
    """Write each distinct sub-agent trace and return a lightweight overview.

    One record is written per distinct ``State``. PDR-style workflows usually
    have one state per attempt/distiller/finalizer; resumed loop workflows may
    share a single state and are written once. Best-effort: an individual trace
    write failure skips that entry but does not fail the workflow run.
    """

    overview: list[dict[str, Any]] = []
    seen: set[int] = set()
    index = 0
    for step in result.steps:
        if id(step.state) in seen:
            continue
        seen.add(id(step.state))
        label = step.role or step.name or f"step{index}"
        safe = "".join(c if c.isalnum() else "_" for c in label)[:40]
        subpath = f"sub/{index:02d}_{safe}.jsonl"
        meta = {
            "workflow": workflow,
            "role": step.role,
            "name": step.name,
            "model": _state_model(step.state),
        }
        try:
            stream, raw = event_stream_bytes(
                run_trace_from_state(
                    state=step.state,
                    trace_id=f"{workflow}.{index:02d}.{label}",
                    producer=f"workflow:{workflow}",
                    meta=meta,
                )
            )
            put(f"out/{subpath}", stream)
            if raw is not None:
                put(f"out/sub/{index:02d}_{safe}.raw.jsonl", raw)
            overview.append(
                {
                    "index": index,
                    "label": label,
                    "role": step.role,
                    "name": step.name,
                    "model": meta["model"],
                    "tokens": state_output_tokens(step.state),
                    "output": step.output or "",
                    "subpath": subpath,
                }
            )
        except Exception:
            pass
        index += 1
    return overview


def compose_workflow_trace_state(
    state: State,
    *,
    overview: list[dict[str, Any]] | None,
    final_output: str,
    agent_name: str,
) -> State | None:
    """Fold workflow sub-traces into a lightweight tree on the facade trace."""

    if not overview:
        return None

    composed = State(task=state.task)
    existing = getattr(state, "data", None)
    if isinstance(existing, dict):
        composed.data.update(existing)
    composed.send("task", "user", agent_name, state.task)
    composed.record_event(AgentStartEvent())
    composed.record_event(TurnStartEvent(agent=agent_name))
    for entry in overview:
        call_id = f"step-{entry['index']:02d}"
        label = str(entry["label"])
        composed.send(
            "thought",
            agent_name,
            agent_name,
            [
                ToolCallBlock(
                    id=call_id,
                    name=label,
                    arguments={"trace": f"out/{entry['subpath']}"},
                )
            ],
        )
        composed.record_event(
            ToolExecutionStartEvent(tool_call_id=call_id, tool_name=label)
        )
        composed.record_event(
            ToolExecutionEndEvent(
                tool_call_id=call_id,
                tool_name=label,
                is_error=False,
                terminate=False,
            )
        )
        composed.send(
            "tool_result",
            "tool",
            agent_name,
            [
                ToolResultBlock(
                    tool_call_id=call_id,
                    tool_name=label,
                    content=(TextBlock(text=workflow_overview_summary(entry)),),
                )
            ],
            role="user",
        )
    composed.record_event(TurnEndEvent(agent=agent_name))
    composed.send("final", agent_name, "user", final_output)
    composed.record_event(AgentEndEvent(reason="done"))
    return composed


def workflow_overview_summary(entry: Mapping[str, Any]) -> str:
    """The text shown when a workflow-tree node is selected in the viewer."""

    head = (
        f"{entry['label']}  ·  model={entry['model'] or 'unknown'}  "
        f"·  ~{entry['tokens']} output tokens\n"
        f"full trace: out/{entry['subpath']}  "
        f"(+ .raw.jsonl sibling for Wire debug)\n"
    )
    body = (entry.get("output") or "").strip()
    if len(body) > 1200:
        body = body[:1200] + "\n… (truncated — open the sub-trace for the full output)"
    return head + ("\n" + body if body else "")


def _state_model(state: State) -> str:
    """The served model recorded on the most recent assistant turn (or '')."""

    for message in reversed(state.messages):
        model = getattr(message, "model", "") or ""
        if model:
            return str(model)
    return ""
