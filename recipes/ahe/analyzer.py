from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.strategy import parse_model_json
from simple_agent_lab.evolution.types import Decision, Run, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

MAX_DECISIONS = 5
MAX_KNOWLEDGE_CHARS = 320
MAX_EVENT_CHARS = 240
MAX_RESULT_CHARS = 600


@dataclass(frozen=True)
class AnalysisResult:
    overview_path: Path
    detail_dir: Path
    index_path: Path
    overview: str
    index: Mapping[str, object]


def analyze_runs(
    provider: Provider,
    runs: Sequence[Run],
    version: Version,
    decisions: Sequence[Decision],
    output_dir: Path,
    knowledge: Sequence[str] = (),
    complete_fn: Callable[[LLMRequest], Any] = complete,
    max_tokens: int = 4000,
) -> AnalysisResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    request = LLMRequest(
        provider=provider,
        messages=[llm_message("user", _build_prompt(version, runs, decisions, knowledge))],
        system_prompt=_system_prompt(),
        max_tokens=max_tokens,
    )
    payload = parse_model_json(complete_fn(request).text)

    run_count = len(runs)
    failed_runs = [run for run in runs if _is_failed(run)]

    overview = _overview_text(payload.get("overview"), version=version, run_count=run_count)
    overview_path = output_dir / "overview.md"
    overview_path.write_text(overview, encoding="utf-8")

    details_written: dict[str, str] = {}
    model_details = payload.get("details")
    if isinstance(model_details, Mapping):
        for run in runs:
            detail_text = model_details.get(run.instance_id)
            if isinstance(detail_text, str):
                path = detail_dir / _safe_detail_filename(run.instance_id)
                path.write_text(detail_text, encoding="utf-8")
                details_written[run.instance_id] = path.relative_to(output_dir).as_posix()

    for run in failed_runs:
        if run.instance_id in details_written:
            continue
        path = detail_dir / _safe_detail_filename(run.instance_id)
        path.write_text(_fallback_detail(run), encoding="utf-8")
        details_written[run.instance_id] = path.relative_to(output_dir).as_posix()

    index_data: dict[str, object] = {
        "version": version.hash,
        "run_count": run_count,
        "failed_count": len(failed_runs),
        "patterns": _patterns_list(payload.get("patterns")),
        "details": details_written,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return AnalysisResult(
        overview_path=overview_path,
        detail_dir=detail_dir,
        index_path=index_path,
        overview=overview,
        index=index_data,
    )


def _system_prompt() -> str:
    return (
        "You are analyzing baseline agent runs for a self-evolution recipe.\n"
        "Return only JSON with keys: overview, details, patterns.\n"
        "The overview should summarize the overall failure modes.\n"
        "The details object may include per-instance markdown notes.\n"
        "Each pattern should explain a recurring root cause and likely component."
    )


def _build_prompt(
    version: Version,
    runs: Sequence[Run],
    decisions: Sequence[Decision],
    knowledge: Sequence[str],
) -> str:
    sections = [
        f"Current version: {version.hash}",
        _decision_section(decisions),
        _knowledge_section(knowledge),
        _runs_section(runs),
        (
            "Return JSON with:\n"
            '{\n  "overview": "# Overview\\n...",\n'
            '  "details": {"i1": "# i1\\n..."},\n'
            '  "patterns": [{"id": "pat-1", "instances": ["i1"], '
            '"likely_component": "tool_implementation", '
            '"root_cause": "..."}]\n}\n'
        ),
    ]
    return "\n\n".join(section for section in sections if section)


def _decision_section(decisions: Sequence[Decision]) -> str:
    if not decisions:
        return "Previous decisions: none."
    lines = ["Previous decisions:"]
    for decision in decisions[:MAX_DECISIONS]:
        lines.append(
            f"- {decision.id}: outcome={decision.outcome}, reason={_clip(decision.reason, MAX_RESULT_CHARS)}"
        )
    if len(decisions) > MAX_DECISIONS:
        lines.append(f"- ... {len(decisions) - MAX_DECISIONS} more decisions omitted")
    return "\n".join(lines)


def _knowledge_section(knowledge: Sequence[str]) -> str:
    if not knowledge:
        return "Knowledge snippets: none."
    lines = ["Knowledge snippets:"]
    for snippet in knowledge:
        lines.append(f"- {_clip(snippet, MAX_KNOWLEDGE_CHARS)}")
    return "\n".join(lines)


def _runs_section(runs: Sequence[Run]) -> str:
    lines = ["Run summaries:"]
    for run in runs:
        keys = ", ".join(sorted(dict(run.result).keys())) or "(none)"
        lines.append(
            "- "
            f"{run.instance_id}: reward={run.reward}, "
            f"result_keys=[{keys}], trajectory={_trajectory_preview(run)}"
        )
    return "\n".join(lines)


def _trajectory_preview(run: Run) -> str:
    events = run.events()
    if not events:
        return "[]"
    preview = [_clip(_event_preview(event), MAX_EVENT_CHARS) for event in events[:2]]
    if len(events) > 2:
        preview.append("...")
    return "[" + ", ".join(preview) + "]"


def _overview_text(value: object, *, version: Version, run_count: int) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return f"# Overview\nAnalyzed version {version.hash} across {run_count} runs.\n"


def _fallback_detail(run: Run) -> str:
    result_json = _clip(
        json.dumps(run.result, indent=2, sort_keys=True, ensure_ascii=True),
        MAX_RESULT_CHARS,
    )
    return (
        f"# Fallback analysis for {run.instance_id}\n\n"
        f"- Run ref: {run.ref}\n"
        f"- Reward: {run.reward}\n\n"
        "## Result JSON\n"
        f"```json\n{result_json}\n```\n\n"
        "## Trajectory Preview\n"
        f"{_trajectory_preview(run)}\n"
    )


def _patterns_list(value: object) -> list[object]:
    if isinstance(value, list):
        patterns = [_normalize_pattern(pattern) for pattern in value]
        return sorted(patterns, key=lambda pattern: str(pattern.get("id", "")))
    return []


def _is_failed(run: Run) -> bool:
    reward = run.reward
    return reward is not None and reward <= 0


def _safe_detail_filename(instance_id: str) -> str:
    slug = re.sub(r"_+", "_", "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in instance_id
    )).strip("_")
    if not slug:
        slug = "instance"
    return slug + ".md"


def _normalize_pattern(pattern: object) -> dict[str, object]:
    if isinstance(pattern, Mapping):
        return dict(pattern)
    return {"id": str(pattern)}


def _event_preview(event: object) -> str:
    try:
        text = json.dumps(event, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        text = repr(event)
    return text


def _clip(text: object, limit: int) -> str:
    value = str(text)
    if limit < 0:
        return value
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."
