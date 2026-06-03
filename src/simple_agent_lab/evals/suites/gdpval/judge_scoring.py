"""Rubric normalization and scoring helpers for GDPVal judge runs."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def normalize_rubrics(raw: Any) -> list[dict[str, Any]]:
    """Return GDPVal rubrics as ``{"index", "weight", "criterion"}`` items."""

    value = _coerce_json_like(raw)
    if isinstance(value, Mapping):
        for key in ("rubrics", "rubric_json", "items"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    rubrics: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        item_dict = dict(item)
        criterion = str(
            item_dict.get("criterion")
            or item_dict.get("rubric_content")
            or item_dict.get("rubric")
            or item_dict.get("description")
            or ""
        ).strip()
        if not criterion:
            continue
        rubrics.append(
            {
                "index": len(rubrics),
                "source_index": index,
                "weight": _safe_float(
                    item_dict.get("score", item_dict.get("weight", 1.0)),
                    default=1.0,
                ),
                "criterion": criterion,
            }
        )
    return rubrics


def parse_judge_payload(raw: Any) -> dict[str, Any]:
    """Parse judge JSON from a dict, a JSON string, or a fenced JSON response."""

    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError("judge payload must be a JSON object or string")
    text = raw.strip()
    if not text:
        raise ValueError("judge payload is empty")

    candidates: list[str] = []
    for fence in re.finditer(
        r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE
    ):
        candidates.append(fence.group(1).strip())
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            for index, char in enumerate(candidate):
                if char not in "{[":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError("judge payload did not contain a JSON object")


def score_judgment(payload: Mapping[str, Any], rubrics: Any) -> dict[str, Any]:
    """Normalize a judge payload and compute weighted GDPVal rubric score."""

    normalized_rubrics = normalize_rubrics(rubrics)
    raw_results = payload.get("rubric_results") or payload.get("rubrics_result") or []
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raw_results = []
    result_items = [dict(item) for item in raw_results if isinstance(item, Mapping)]

    max_score = sum(
        max(0.0, _safe_float(item["weight"])) for item in normalized_rubrics
    )
    scored: list[dict[str, Any]] = []
    earned_total = 0.0
    for position, rubric in enumerate(normalized_rubrics):
        result = _match_result(rubric, position, result_items)
        weight = max(0.0, _safe_float(rubric["weight"]))
        grade = _grade_from_result(result)
        earned = weight * grade
        earned_total += earned
        scored.append(
            {
                "index": rubric["index"],
                "weight": weight,
                "criterion": rubric["criterion"],
                "passed": grade >= 0.999,
                "grade": grade,
                "earned": earned,
                "explanation": str(
                    result.get("explanation")
                    or result.get("grade_explanation")
                    or result.get("reason")
                    or ""
                ).strip(),
            }
        )

    score = earned_total / max_score if max_score > 0 else 0.0
    status = "judged" if normalized_rubrics else "no_rubrics"
    return {
        "status": status,
        "score": score,
        "earned_score": earned_total,
        "max_score": max_score,
        "rubric_results": scored,
        "overall_explanation": str(
            payload.get("overall_explanation")
            or payload.get("rationale")
            or payload.get("explanation")
            or ""
        ).strip(),
    }


def _coerce_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def _match_result(
    rubric: Mapping[str, Any], position: int, results: list[dict[str, Any]]
) -> dict[str, Any]:
    for item in results:
        if item.get("index") == rubric.get("index"):
            return item
    criterion = str(rubric.get("criterion") or "").strip()
    if criterion:
        for item in results:
            if str(item.get("criterion") or "").strip() == criterion:
                return item
    if position < len(results):
        return results[position]
    return {}


def _grade_from_result(result: Mapping[str, Any]) -> float:
    for key in ("grade", "grade_B", "grade_b", "satisfied", "passed", "pass"):
        if key in result:
            return _clamp(_boolish_float(result.get(key)))
    if "earned" in result and "weight" in result:
        weight = _safe_float(result.get("weight"), default=0.0)
        if weight > 0:
            return _clamp(_safe_float(result.get("earned")) / weight)
    if "earned" in result:
        return _clamp(_safe_float(result.get("earned")))
    return 0.0


def _boolish_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "pass", "passed", "1", "y"}:
        return 1.0
    if text in {"false", "no", "fail", "failed", "0", "n", ""}:
        return 0.0
    return _safe_float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
