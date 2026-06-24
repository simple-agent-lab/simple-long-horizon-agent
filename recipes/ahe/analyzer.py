from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.strategy import parse_model_json
from simple_agent_lab.evolution.types import Decision, Run, RunScores, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

MAX_DECISIONS = 5
MAX_RUNS_IN_PROMPT = 8
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
    run_scores: RunScores | None = None,
    complete_fn: Callable[[LLMRequest], Any] = complete,
    max_tokens: int = 4000,
) -> AnalysisResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    request = LLMRequest(
        provider=provider,
        messages=[
            llm_message(
                "user",
                _build_prompt(
                    version, runs, decisions, knowledge, run_scores=run_scores
                ),
            )
        ],
        system_prompt=_system_prompt(),
        max_tokens=max_tokens,
    )
    try:
        payload = parse_model_json(complete_fn(request).text)
        parse_error = ""
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        payload = {}
        parse_error = f"{type(exc).__name__}: {exc}"

    run_count = len(runs)
    failed_runs = [run for run in runs if _is_failed(run, run_scores=run_scores)]

    overview = _overview_text(
        payload.get("overview"), version=version, run_count=run_count
    )
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
                details_written[run.instance_id] = path.relative_to(
                    output_dir
                ).as_posix()

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
    if parse_error:
        index_data["analyzer_error"] = parse_error
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
    *,
    run_scores: RunScores | None = None,
) -> str:
    prompt_runs = _prompt_runs(runs, run_scores=run_scores)
    total_failed = sum(1 for run in runs if _is_failed(run, run_scores=run_scores))
    sections = [
        f"Current version: {version.hash}",
        _decision_section(decisions),
        _knowledge_section(knowledge),
        _runs_section(
            prompt_runs,
            total_runs=len(runs),
            total_failed=total_failed,
            run_scores=run_scores,
        ),
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


def _runs_section(
    runs: Sequence[Run],
    *,
    total_runs: int,
    total_failed: int,
    run_scores: RunScores | None = None,
) -> str:
    lines = []
    passed_count = total_runs - total_failed
    lines.append(
        f"Showing {len(runs)} of {total_runs} runs; failed={total_failed} passed={passed_count}"
    )
    for run in runs:
        keys = ", ".join(_result_keys(run.result)) or "(none)"
        lines.append(
            f"- {run.instance_id}: reward={_reward(run, run_scores)}, "
            f"result_keys=[{_clip(keys, 120)}], trajectory={_trajectory_preview(run)}"
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
    result = dict(run.result)
    result_keys = _result_keys(result)
    selected = {
        key: result[key] for key in _selected_result_keys(result) if key in result
    }
    result_preview = _clip(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True),
        MAX_RESULT_CHARS,
    )
    return (
        f"# Fallback analysis for {run.instance_id}\n\n"
        f"- Run ref: {run.ref}\n"
        f"- Reward: {run.reward}\n\n"
        f"- Result keys: {', '.join(result_keys) if result_keys else '(none)'}\n\n"
        "## Selected Result Fields\n"
        f"{_format_selected_fields(selected)}\n\n"
        "## Result JSON Preview\n"
        f"```json\n{result_preview}\n```\n\n"
        "## Trajectory Preview\n"
        f"{_trajectory_preview(run)}\n"
    )


def _patterns_list(value: object) -> list[object]:
    if isinstance(value, list):
        patterns = [_normalize_pattern(pattern) for pattern in value]
        return sorted(patterns, key=lambda pattern: str(pattern.get("id", "")))
    return []


def _is_failed(run: Run, *, run_scores: RunScores | None = None) -> bool:
    reward = _reward(run, run_scores)
    return reward is not None and reward <= 0


def _reward(run: Run, run_scores: RunScores | None = None) -> float | None:
    if run_scores is not None and run.instance_id in run_scores:
        return float(run_scores[run.instance_id].get("reward", 0.0))
    return run.reward


def _safe_detail_filename(instance_id: str) -> str:
    slug = re.sub(
        r"_+",
        "_",
        "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in instance_id),
    ).strip("_")
    if not slug:
        slug = "instance"
    return slug + ".md"


def _normalize_pattern(pattern: object) -> dict[str, object]:
    if isinstance(pattern, Mapping):
        return dict(pattern)
    return {"id": str(pattern)}


def _event_preview(event: object) -> str:
    try:
        text = json.dumps(
            event, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
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


def _prompt_runs(
    runs: Sequence[Run], *, run_scores: RunScores | None = None
) -> list[Run]:
    failed = [run for run in runs if _is_failed(run, run_scores=run_scores)]
    passed = [run for run in runs if not _is_failed(run, run_scores=run_scores)]
    selected = list(failed[:MAX_RUNS_IN_PROMPT])
    if len(selected) < MAX_RUNS_IN_PROMPT:
        selected.extend(passed[: MAX_RUNS_IN_PROMPT - len(selected)])
    return selected


def _result_keys(result: Mapping[str, object]) -> list[str]:
    return sorted(str(key) for key in result.keys())


def _selected_result_keys(result: Mapping[str, object]) -> tuple[str, ...]:
    keys = ("resolved", "score", "error", "message", "agent_package")
    return tuple(key for key in keys if key in result)


def _format_selected_fields(selected: Mapping[str, object]) -> str:
    if not selected:
        return "- none"
    lines = []
    for key in ("resolved", "score", "error", "message", "agent_package"):
        if key in selected:
            lines.append(f"- {key}: {_clip(selected[key], MAX_RESULT_CHARS)}")
    return "\n".join(lines)
