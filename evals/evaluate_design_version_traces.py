"""Evaluate design-version trajectories without running the demos.

Run from the repo root:

    PYTHONPATH=src python3 evals/evaluate_design_version_traces.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.evaluation import EvalResult, eval_result_record
from simple_agent_lab.trajectory import read_jsonl, write_jsonl


def evaluate(record: dict[str, Any]) -> EvalResult:
    trace_id = str(record["trace_id"])
    if trace_id == "01_functional_loop.shopping_helper":
        return evaluate_01(record)
    if trace_id == "02_balanced_runtime.agent_as_tool":
        return evaluate_02(record)
    if trace_id == "03_event_runtime.weather_graph":
        return evaluate_03(record)
    return EvalResult(
        trace_id=trace_id,
        scorer="design_version.unknown",
        passed=False,
        score=0.0,
        reason="No scorer registered for this trace_id.",
        metrics={},
    )


def evaluate_01(record: dict[str, Any]) -> EvalResult:
    messages = list(record.get("messages") or [])
    final = _last_message_text(messages, sender="assistant", kind="final")
    tool_errors = sum(
        1
        for message in messages
        if message.get("role") == "tool_result" and message.get("is_error")
    )
    tool_calls = sum(
        len(turn.get("output_message", {}).get("tool_calls") or [])
        for turn in record.get("model_turns") or []
    )
    passed = "$139.93" in final and tool_calls >= 1 and tool_errors == 0
    return EvalResult(
        trace_id=str(record["trace_id"]),
        scorer="design_version.shopping_total.v1",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="final total uses calculate tool" if passed else "missing expected total or tool use",
        metrics={
            "model_calls": len(record.get("model_turns") or []),
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "messages": len(messages),
            "final": final,
        },
    )


def evaluate_02(record: dict[str, Any]) -> EvalResult:
    messages = list(record.get("messages") or [])
    events = list(record.get("events") or [])
    final = _last_message_text(messages, sender="coordinator", kind="final")
    tool_ends = [event for event in events if event.get("kind") == "tool_execution_end"]
    tool_errors = sum(1 for event in tool_ends if event.get("is_error"))
    child_counts = [
        int((message.get("data", {}).get("details") or {}).get("child_event_count", 0))
        for message in messages
        if message.get("role") == "tool_result" and message.get("tool_name") == "run_agent"
    ]
    passed = bool(final) and len(tool_ends) >= 1 and tool_errors == 0 and max(child_counts, default=0) > 0
    return EvalResult(
        trace_id=str(record["trace_id"]),
        scorer="design_version.agent_as_tool.v1",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="coordinator delegated to child agent" if passed else "missing delegation evidence",
        metrics={
            "model_calls": len(record.get("model_turns") or []),
            "tool_calls": len(tool_ends),
            "tool_errors": tool_errors,
            "messages": len(messages),
            "child_event_counts": child_counts,
            "final": final,
        },
    )


def evaluate_03(record: dict[str, Any]) -> EvalResult:
    messages = list(record.get("messages") or [])
    events = list(record.get("events") or [])
    final = _last_message_text(messages, sender="travel_advisor", kind="final")
    graph_end = next((event for event in reversed(events) if event.get("kind") == "graph_end"), {})
    graph_path = list(graph_end.get("data", {}).get("path") or graph_end.get("path") or [])
    tool_ends = [event for event in events if event.get("kind") == "tool_execution_end"]
    tool_errors = sum(1 for event in tool_ends if (event.get("data") or event).get("is_error"))
    passed = (
        graph_path == ["weather_researcher", "travel_advisor"]
        and len(tool_ends) >= 1
        and tool_errors == 0
        and "jacket" in final.lower()
    )
    return EvalResult(
        trace_id=str(record["trace_id"]),
        scorer="design_version.weather_graph.v1",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="weather graph reaches advisor with clothing answer" if passed else "missing graph path or final advice",
        metrics={
            "model_calls": len(record.get("model_turns") or []),
            "tool_calls": len(tool_ends),
            "tool_errors": tool_errors,
            "messages": len(messages),
            "graph_path": graph_path,
            "final": final,
        },
    )


def _last_message_text(messages: list[dict[str, Any]], *, sender: str, kind: str) -> str:
    for message in reversed(messages):
        if message.get("sender") == sender and message.get("kind") == kind:
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        default=str(ROOT / "evals/out/design_version_trajectories.jsonl"),
        help="Trajectory JSONL input.",
    )
    parser.add_argument(
        "--jsonl",
        default=str(ROOT / "evals/out/design_version_eval_results.jsonl"),
        help="Eval-result JSONL output.",
    )
    args = parser.parse_args()

    trace_records = [
        record
        for record in read_jsonl(args.traces)
        if record.get("type") == "trajectory"
    ]
    results = [evaluate(record) for record in trace_records]
    write_jsonl(args.jsonl, [eval_result_record(result) for result in results])

    print(f"wrote {len(results)} eval results to {args.jsonl}")
    for result in results:
        status = "pass" if result.passed else "fail"
        print(
            f"{result.trace_id}: {status} "
            f"score={result.score} model_calls={result.metrics.get('model_calls')}"
        )


if __name__ == "__main__":
    main()
