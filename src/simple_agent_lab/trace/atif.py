"""Export Simple Agent Lab traces to Harbor ATIF dictionaries.

Harbor expects installed agents to write `/logs/agent/trajectory.json` in the
Agent Trajectory Interchange Format. The native Simple Agent Lab trace remains
the richer source of truth; this module provides a small lossy projection for
Harbor metrics and training/export tooling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..messages import TokenUsage
from ..model_metadata import RunCost
from .jsonl import json_safe


ATIF_SCHEMA_VERSION = "ATIF-v1.7"


def atif_trajectory_from_run(
    *,
    trace_id: str,
    task: str,
    events: Iterable[Any],
    messages: Iterable[Any] = (),
    agent_name: str,
    agent_version: str,
    model_name: str,
    producer: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a Harbor ATIF trajectory from a Simple Agent Lab run.

    The conversion keeps the turn-level shape Harbor needs: user task, assistant
    turns, tool calls, observations, and aggregate token/cost metrics. The
    native v5 JSONL trace should be written alongside this ATIF file for full
    replay/debugging detail.
    """

    event_list = list(events)
    message_list = list(messages)
    run_cost = RunCost.from_run(event_list, message_list)
    steps: list[dict[str, Any]] = []
    next_step_id = 1

    system_prompt = _first_system_prompt(event_list)
    if system_prompt:
        steps.append(
            {
                "step_id": next_step_id,
                "source": "system",
                "message": system_prompt,
            }
        )
        next_step_id += 1

    steps.append({"step_id": next_step_id, "source": "user", "message": task})
    next_step_id += 1

    pending_tool_steps: dict[str, dict[str, Any]] = {}
    for event in event_list:
        if _kind(event) != "message":
            continue
        message = _value(event, "message")
        role = _value(message, "role")
        if role == "assistant":
            step = _assistant_step(
                message,
                step_id=next_step_id,
                default_model_name=model_name,
            )
            next_step_id += 1
            steps.append(step)
            for tool_call in step.get("tool_calls", []):
                tool_call_id = tool_call.get("tool_call_id")
                if tool_call_id:
                    pending_tool_steps[str(tool_call_id)] = step
        elif _value(message, "kind") == "tool_result":
            for result in _observation_results(message):
                call_id = str(result.get("source_call_id") or "")
                parent = pending_tool_steps.get(call_id)
                if parent is None:
                    continue
                observation = parent.setdefault("observation", {"results": []})
                observation["results"].append(result)

    trajectory_extra = dict(extra or {})
    trajectory_extra.setdefault("source_schema", "simple-agent-lab.trajectory.v5")
    trajectory_extra.setdefault("producer", producer)

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": trace_id,
        "trajectory_id": trace_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
            "extra": trajectory_extra,
        },
        "steps": steps,
        "final_metrics": _final_metrics(run_cost, len(steps)),
        "extra": trajectory_extra,
    }


def _assistant_step(
    message: Any,
    *,
    step_id: int,
    default_model_name: str,
) -> dict[str, Any]:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in _content_blocks(message):
        kind = _value(block, "kind")
        if kind == "text":
            text = _value(block, "text")
            if text:
                text_parts.append(str(text))
        elif kind == "thinking":
            text = _value(block, "text")
            if text:
                thinking_parts.append(str(text))
        elif kind == "tool_call":
            tool_calls.append(
                {
                    "tool_call_id": str(_value(block, "id") or ""),
                    "function_name": str(_value(block, "name") or ""),
                    "arguments": json_safe(_value(block, "arguments") or {}),
                }
            )

    usage = _coerce_usage(_value(message, "usage"))
    step: dict[str, Any] = {
        "step_id": step_id,
        "source": "agent",
        "model_name": str(_value(message, "model") or default_model_name),
        "message": "\n".join(text_parts),
        "llm_call_count": 1 if usage is not None else None,
    }
    if thinking_parts:
        step["reasoning_content"] = "\n".join(thinking_parts)
    if tool_calls:
        step["tool_calls"] = tool_calls
    if usage is not None:
        step["metrics"] = _metrics(usage)
    return {key: value for key, value in step.items() if value is not None}


def _observation_results(message: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in _content_blocks(message):
        if _value(block, "kind") != "tool_result":
            continue
        result: dict[str, Any] = {
            "source_call_id": str(_value(block, "tool_call_id") or ""),
            "content": _visible_text(_value(block, "content") or ()),
            "extra": {
                "is_error": bool(_value(block, "is_error") or False),
                "tool_name": str(_value(block, "tool_name") or ""),
            },
        }
        results.append(result)
    return results


def _metrics(usage: TokenUsage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.input_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens,
        "completion_tokens": usage.output_tokens,
        "cached_tokens": usage.cache_read_tokens,
        "cost_usd": 0.0,
        "extra": {"cache_write_tokens": usage.cache_write_tokens},
    }


def _final_metrics(run_cost: RunCost, total_steps: int) -> dict[str, Any]:
    tokens = run_cost.total_tokens
    return {
        "total_prompt_tokens": tokens.input_tokens
        + tokens.cache_read_tokens
        + tokens.cache_write_tokens,
        "total_completion_tokens": tokens.output_tokens,
        "total_cached_tokens": tokens.cache_read_tokens,
        "total_cost_usd": round(run_cost.total_usd, 6),
        "total_steps": total_steps,
        "extra": {
            "cache_write_tokens": tokens.cache_write_tokens,
            "llm_call_count": run_cost.calls,
            "unpriced_models": list(run_cost.unpriced_models),
        },
    }


def _first_system_prompt(events: Sequence[Any]) -> str:
    for event in events:
        if _kind(event) != "agent_start":
            continue
        prompt = _value(event, "system_prompt")
        if prompt:
            return str(prompt)
    return ""


def _content_blocks(message: Any) -> Sequence[Any]:
    content = _value(message, "content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return content
    return ()


def _visible_text(blocks: Any) -> str:
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return str(blocks)
    texts: list[str] = []
    for block in blocks:
        if _value(block, "kind") == "text":
            text = _value(block, "text")
            if text:
                texts.append(str(text))
    return "\n".join(texts)


def _coerce_usage(value: Any) -> TokenUsage | None:
    if isinstance(value, TokenUsage):
        return value if value.context_tokens > 0 else None
    if not isinstance(value, Mapping):
        return None
    usage = TokenUsage(
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        cache_read_tokens=int(value.get("cache_read_tokens", 0)),
        cache_write_tokens=int(value.get("cache_write_tokens", 0)),
    )
    return usage if usage.context_tokens > 0 else None


def _kind(event: Any) -> str:
    kind = _value(event, "kind")
    return str(getattr(kind, "value", kind) or "")


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


__all__ = ["ATIF_SCHEMA_VERSION", "atif_trajectory_from_run"]
