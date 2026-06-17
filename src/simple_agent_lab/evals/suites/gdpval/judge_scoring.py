"""Rubric normalization and scoring helpers for GDPVal judge runs."""

from __future__ import annotations

import ast
import html
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

    tagged = _extract_tag_json(text, "judge_result")
    if isinstance(tagged, Mapping):
        return dict(tagged)

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


def parse_gsb_judge_payload(raw: Any) -> dict[str, Any]:
    """Parse a GDPVal GSB judge payload from JSON or XML-tagged JSON chunks."""

    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError("GSB judge payload must be a JSON object or string")
    text = raw.strip()
    if not text:
        raise ValueError("GSB judge payload is empty")

    try:
        return parse_judge_payload(text)
    except ValueError:
        pass

    direction = parse_gsb_direction_payload(text)
    if direction:
        return {"reverse": direction}
    raise ValueError("GSB judge payload did not contain JSON or GSB XML tags")


def parse_gsb_direction_payload(raw: Any) -> dict[str, Any]:
    """Parse one swalm-style GDPVal GSB direction output."""

    if isinstance(raw, Mapping):
        payload = dict(raw)
        if "rubrics_result" in payload or "rubric_results" in payload:
            return {
                "rubrics_result": _normalize_direction_rubrics(
                    payload.get("rubrics_result") or payload.get("rubric_results")
                ),
                "overall": _normalize_direction_overall(
                    payload.get("overall") or payload
                ),
            }
        return {}
    if not isinstance(raw, str):
        raise ValueError("GSB direction payload must be a JSON object or string")
    text = raw.strip()
    if not text:
        raise ValueError("GSB direction payload is empty")

    try:
        payload = parse_judge_payload(text)
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        if "rubrics_result" in payload or "rubric_results" in payload:
            return {
                "rubrics_result": _normalize_direction_rubrics(
                    payload.get("rubrics_result") or payload.get("rubric_results")
                ),
                "overall": _normalize_direction_overall(
                    payload.get("overall") or payload
                ),
            }
        if "overall" in payload:
            return {
                "rubrics_result": [],
                "overall": _normalize_direction_overall(payload.get("overall")),
            }

    rubrics = _parse_rubrics_result(text)
    overall = _parse_overall(text)
    if rubrics is None and overall is None:
        return {}
    return {
        "rubrics_result": rubrics or [],
        "overall": overall or {},
    }


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


def score_gsb_judgment(payload: Mapping[str, Any], rubrics: Any) -> dict[str, Any]:
    """Compute GDPVal GSB scores from forward/reverse A/B judgments."""

    normalized_rubrics = normalize_rubrics(rubrics)
    max_score = sum(
        max(0.0, _safe_float(item["weight"])) for item in normalized_rubrics
    )
    reverse_payload = _direction_payload(payload, "reverse")
    forward_payload = _direction_payload(payload, "forward")

    reverse = _score_gsb_direction(
        reverse_payload,
        normalized_rubrics,
        criterion_map={"A>B": -1.0, "A=B": 0.0, "A<B": 1.0},
        overall_map={
            "A>>B": 0.0,
            "A>B": 0.25,
            "A=B": 0.5,
            "A<B": 0.75,
            "A<<B": 1.0,
        },
    )
    forward = _score_gsb_direction(
        forward_payload,
        normalized_rubrics,
        criterion_map={"A>B": 1.0, "A=B": 0.0, "A<B": -1.0},
        overall_map={
            "A>>B": 1.0,
            "A>B": 0.75,
            "A=B": 0.5,
            "A<B": 0.25,
            "A<<B": 0.0,
        },
    )

    has_reverse_rubrics = bool(reverse["parsed_payload"]["rubrics_result"])
    has_forward_rubrics = bool(forward["parsed_payload"]["rubrics_result"])
    has_any_rubrics = has_reverse_rubrics or has_forward_rubrics
    reverse_rubric_raw = (
        reverse["rubrics_weighted_score"]
        if reverse["rubrics_weighted_score"] is not None
        else 0.0
    )
    forward_rubric_raw = (
        forward["rubrics_weighted_score"]
        if forward["rubrics_weighted_score"] is not None
        else 0.0
    )
    reverse_overall_raw = (
        reverse["llm_gsb_score"] if reverse["llm_gsb_score"] is not None else 0.0
    )
    forward_overall_raw = (
        forward["llm_gsb_score"] if forward["llm_gsb_score"] is not None else 0.0
    )
    combined_raw = (reverse_rubric_raw + forward_rubric_raw) / 2.0
    llm_raw = (reverse_overall_raw + forward_overall_raw) / 2.0
    combined_weighted_score = (
        _three_way_score(combined_raw, high=0.02, low=-0.02)
        if has_any_rubrics
        else None
    )
    llm_score = _three_way_score(llm_raw, high=0.55, low=0.45)

    score_a_forward = forward["score_a"]
    score_b_forward = forward["score_b"]
    score_a_reverse = reverse["score_a"]
    score_b_reverse = reverse["score_b"]
    dcg_winrate_raw = (
        score_a_forward + score_b_reverse - score_b_forward - score_a_reverse
    ) / 2.0
    dcg_winrate = _three_way_score(dcg_winrate_raw, high=0.02, low=-0.02)
    score_process = (
        score_b_reverse
        if has_reverse_rubrics
        else score_a_forward
        if has_forward_rubrics
        else 0.0
    )
    fallback_score = (
        reverse["llm_gsb_score"]
        if reverse["llm_gsb_score"] is not None
        else forward["llm_gsb_score"]
        if forward["llm_gsb_score"] is not None
        else 0.0
    )
    score = (
        combined_weighted_score
        if combined_weighted_score is not None
        else fallback_score
    )

    status = "gsb_judged" if normalized_rubrics else "no_rubrics"
    return {
        "status": status,
        "score": score,
        "earned_score": score * max_score,
        "max_score": max_score,
        "combined_weighted_score": combined_weighted_score,
        "combined_weighted_score_raw": combined_raw,
        "llm_score": llm_score,
        "llm_score_raw": llm_raw,
        "score_process": score_process,
        "score_a_forward": score_a_forward,
        "score_b_forward": score_b_forward,
        "score_a_reverse": score_a_reverse,
        "score_b_reverse": score_b_reverse,
        "dcg_winrate_raw": dcg_winrate_raw,
        "dcg_winrate": dcg_winrate,
        "rubrics_weighted_score_reverse": reverse["rubrics_weighted_score"],
        "rubrics_weighted_score_forward": forward["rubrics_weighted_score"],
        "llm_gsb_score_reverse": reverse["llm_gsb_score"],
        "llm_gsb_score_forward": forward["llm_gsb_score"],
        "rubrics_score_list_reverse": reverse["rubrics_score_list"],
        "rubrics_score_list_forward": forward["rubrics_score_list"],
        "rubric_results_reverse": reverse["rubric_results"],
        "rubric_results_forward": forward["rubric_results"],
        "overall_explanation_reverse": reverse["overall_explanation"],
        "overall_explanation_forward": forward["overall_explanation"],
        "final_gsb_reverse": reverse["final_gsb"],
        "final_gsb_forward": forward["final_gsb"],
        "rm_eval_result": {
            "reverse": reverse["parsed_payload"],
            "forward": forward["parsed_payload"],
        },
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


def _extract_tag_json(text: str, tag: str) -> Any:
    pattern = rf"<{re.escape(tag)}>\s*([\s\S]*?)\s*</{re.escape(tag)}>"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    body = match.group(1).strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tag {tag!r} did not contain valid JSON") from exc


def _parse_rubrics_result(text: str) -> list[dict[str, Any]] | None:
    match = re.search(
        r"<rubrics_result\b[^>]*>([\s\S]*?)(?:</rubrics_result>|<overall\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        body = match.group(1).strip()
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            normalized = _normalize_direction_rubrics(parsed)
            if normalized:
                return normalized
            xmlish = _parse_xmlish_rubrics_result(body)
            if xmlish is not None:
                return xmlish

    try:
        payload = parse_judge_payload(text)
    except ValueError:
        return _parse_xmlish_rubrics_result(text)
    if isinstance(payload, Mapping):
        return _normalize_direction_rubrics(
            payload.get("rubrics_result") or payload.get("rubric_results")
        )
    return _normalize_direction_rubrics(payload)


def _parse_overall(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"<overall\b[^>]*>([\s\S]*?)</overall>",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        body = match.group(1).strip()
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            normalized = _normalize_direction_overall(parsed)
            if normalized:
                return normalized
            xmlish = _parse_xmlish_overall(body)
            if xmlish:
                return xmlish
    try:
        payload = parse_judge_payload(text)
    except ValueError:
        return None
    if isinstance(payload, Mapping):
        return _normalize_direction_overall(payload.get("overall") or payload)
    return None


def _parse_xmlish_rubrics_result(text: str) -> list[dict[str, Any]] | None:
    blocks = [
        match.group(0)
        for match in re.finditer(
            r"<rubric\b[^>]*/>|<rubric\b[^>]*>[\s\S]*?</rubric>",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if not blocks:
        return None
    items: list[dict[str, Any]] = []
    for block in blocks:
        grade_b = _xml_first_value(block, ("grade_B", "grade_b", "gradeB"))
        pass_value = _xml_first_value(block, ("pass", "passed", "satisfied"))
        grade_b_value = (
            _boolish_grade(grade_b) if grade_b else _boolish_grade(pass_value)
        )
        items.append(
            {
                "score": _safe_float(_xml_first_value(block, ("score",)), default=1.0),
                "criterion": _xml_first_value(block, ("criterion", "rubric_id")),
                "grade_A": _safe_float(
                    _xml_first_value(block, ("grade_A", "grade_a", "gradeA")),
                    default=0.0,
                ),
                "grade_B": 0.0 if grade_b_value is None else grade_b_value,
                "gsb": _xml_first_value(block, ("gsb",)),
                "grade_explanation": _xml_first_value(
                    block,
                    ("grade_explanation", "explanation", "reason"),
                ),
            }
        )
    return _normalize_direction_rubrics(items)


def _parse_xmlish_overall(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    explanation = _xml_tag_text(text, "overall_explanation")
    final_score = _safe_optional_float(_xml_tag_text(text, "final_score"))
    final_gsb = _xml_tag_text(text, "final_gsb")
    if explanation:
        result["overall_explanation"] = explanation
    if final_score is not None:
        result["final_score"] = final_score
    if final_gsb:
        result["final_gsb"] = final_gsb
    return result


def _normalize_direction_rubrics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _normalize_direction_overall(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _xml_attr(block: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(['\"])([\s\S]*?)\1",
        block,
        flags=re.IGNORECASE,
    )
    return html.unescape(match.group(2).strip()) if match else ""


def _xml_tag_text(block: str, name: str) -> str:
    match = re.search(
        rf"<{re.escape(name)}\b[^>]*>([\s\S]*?)</{re.escape(name)}>",
        block,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1)).strip())


def _xml_first_value(block: str, names: tuple[str, ...]) -> str:
    for name in names:
        value = _xml_attr(block, name) or _xml_tag_text(block, name)
        if value:
            return value
    return ""


def _boolish_grade(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "pass", "passed", "1", "y"}:
        return 1.0
    if text in {"false", "no", "fail", "failed", "0", "n"}:
        return 0.0
    parsed = _safe_optional_float(value)
    if parsed is None:
        return None
    return _clamp(parsed)


def _safe_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _direction_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    if key == "reverse" and (
        "rubrics_result" in payload
        or "rubric_results" in payload
        or "overall" in payload
        or "final_gsb" in payload
    ):
        return dict(payload)
    return {}


def _score_gsb_direction(
    payload: Mapping[str, Any],
    rubrics: list[dict[str, Any]],
    *,
    criterion_map: Mapping[str, float],
    overall_map: Mapping[str, float],
) -> dict[str, Any]:
    raw_results = payload.get("rubrics_result") or payload.get("rubric_results") or []
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raw_results = []
    result_items = [dict(item) for item in raw_results if isinstance(item, Mapping)]

    rubric_results: list[dict[str, Any]] = []
    weighted_gsb_total = 0.0
    score_a_total = 0.0
    score_b_total = 0.0
    scored_weight_total = 0.0
    valid_gsb_count = 0
    for position, result in enumerate(result_items):
        rubric = _rubric_for_gsb_result(result, position, rubrics)
        weight = max(0.0, _safe_float(rubric["weight"]))
        gsb = _normalize_gsb(result.get("gsb"))
        gsb_score = criterion_map.get(gsb, 0.0)
        if gsb in criterion_map:
            valid_gsb_count += 1
        grade_a = _grade_for_key(result, "grade_A", "grade_a")
        grade_b = _grade_for_key(result, "grade_B", "grade_b")
        scored_weight_total += weight
        weighted_gsb_total += weight * gsb_score
        score_a_total += weight * grade_a
        score_b_total += weight * grade_b
        rubric_results.append(
            {
                "index": rubric["index"],
                "weight": weight,
                "criterion": rubric["criterion"],
                "grade_A": grade_a,
                "grade_B": grade_b,
                "gsb": gsb,
                "gsb_score": gsb_score,
                "explanation": str(
                    result.get("grade_explanation")
                    or result.get("explanation")
                    or result.get("reason")
                    or ""
                ).strip(),
            }
        )

    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        overall = payload
    final_gsb = _normalize_gsb(overall.get("final_gsb") if overall else None)
    llm_gsb_score = overall_map.get(final_gsb)
    return {
        "parsed_payload": {
            "rubrics_result": result_items,
            "overall": dict(overall) if isinstance(overall, Mapping) else {},
        },
        "rubric_results": rubric_results,
        "rubrics_score_list": [
            {
                "index": item["index"],
                "weight": item["weight"],
                "gsb": item["gsb"],
                "gsb_score": item["gsb_score"],
            }
            for item in rubric_results
        ],
        "rubrics_weighted_score": (
            weighted_gsb_total / scored_weight_total
            if scored_weight_total > 0 and valid_gsb_count
            else None
        ),
        "score_a": (
            score_a_total / scored_weight_total if scored_weight_total > 0 else 0.0
        ),
        "score_b": (
            score_b_total / scored_weight_total if scored_weight_total > 0 else 0.0
        ),
        "final_gsb": final_gsb,
        "llm_gsb_score": llm_gsb_score,
        "overall_explanation": str(
            overall.get("overall_explanation")
            or overall.get("explanation")
            or overall.get("rationale")
            or ""
        ).strip()
        if isinstance(overall, Mapping)
        else "",
    }


def _rubric_for_gsb_result(
    result: Mapping[str, Any], position: int, rubrics: list[dict[str, Any]]
) -> dict[str, Any]:
    for key in ("index", "rubric_index", "source_index"):
        if key not in result:
            continue
        wanted = int(_safe_float(result.get(key), default=-1.0))
        for rubric in rubrics:
            if wanted in {int(rubric["index"]), int(rubric.get("source_index", -1))}:
                return rubric
    if 0 <= position < len(rubrics):
        return rubrics[position]
    return {
        "index": position,
        "source_index": position,
        "weight": _safe_float(result.get("score", result.get("weight", 1.0))),
        "criterion": str(result.get("criterion") or ""),
    }


def _grade_for_key(result: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in result:
            return _clamp(_boolish_float(result.get(key)))
    return 0.0


def _normalize_gsb(value: Any) -> str:
    text = str(value or "").upper()
    return re.sub(r"\s+", "", text)


def _three_way_score(value: float | None, *, high: float, low: float) -> float:
    if value is None:
        return 0.0
    if value > high:
        return 1.0
    if value < low:
        return 0.0
    return 0.5


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
