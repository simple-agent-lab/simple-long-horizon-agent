"""Read persisted dynamic-workflow artifacts for eval result payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_workflow_artifacts(
    root: Path, *, embed_traces_in_calls: bool = True
) -> dict[str, Any]:
    """Return the saved script, journal, result, calls, and subagent traces."""

    if not root.exists():
        return {}
    result = _read_json(root / "workflow_result.json")
    journal = _read_jsonl(root / "workflow_journal.jsonl")
    script = _read_text(root / "workflow.js")
    traces = _read_subagent_traces(root)
    if embed_traces_in_calls:
        agent_calls = list((result or {}).get("agent_calls") or [])
        for call in agent_calls:
            if not isinstance(call, dict):
                continue
            trace = traces.get(str(call.get("call_id") or ""))
            if trace:
                call["trace"] = trace
    else:
        agent_calls = [
            dict(call)
            for call in ((result or {}).get("agent_calls") or [])
            if isinstance(call, dict)
        ]
    return {
        "workflow_js": script,
        "result": result,
        "journal": journal,
        "agent_calls": agent_calls,
        "subagent_traces": traces,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _read_subagent_traces(root: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    subagents = root / "subagents"
    if not subagents.exists():
        return traces
    for trace_path in subagents.glob("*/trajectory.jsonl"):
        records = _read_jsonl(trace_path)
        if records:
            header = dict(records[0])
            if header.get("type") == "trajectory" and len(records) > 1:
                header["events"] = records[1:]
            traces[trace_path.parent.name] = header
    return traces
