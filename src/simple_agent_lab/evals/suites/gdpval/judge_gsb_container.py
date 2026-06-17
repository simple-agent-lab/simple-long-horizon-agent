"""GDPVal GSB judge container half.

The GSB judge is a second-stage run. It receives candidate deliverables, gold
deliverables, references, and rubrics, then writes a strict JSON judgment with
forward and reverse A/B comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.messages import text_of
from simple_agent_lab.protocols import ToolExecutionStartEvent
from simple_agent_lab.state import State

from .judge_container import (
    _candidate_summary,
    _input_dir_for,
    prepare as prepare,
)
from .judge_gsb_prompts import (
    GDPVAL_GSB_JUDGE_EXCEL_HANDLING_PROMPT,
    GDPVAL_GSB_JUDGE_SYSTEM_PROMPT,
)
from .judge_mcp import gdpval_judge_agent_context
from .judge_scoring import (
    normalize_rubrics,
    parse_gsb_direction_payload,
    parse_gsb_judge_payload,
    score_gsb_judgment,
)

JUDGE_RESULT_FILE = "_gdpval_gsb_judge_result.json"
JUDGE_ATTEMPTS_FILE = "_gdpval_gsb_judge_attempts.json"
DEFAULT_GSB_DIRECTION_ATTEMPTS = 1
MAX_GSB_DIRECTION_ATTEMPTS = 5
GSB_DIRECTION_ATTEMPTS_ENV = "GDPVAL_GSB_DIRECTION_ATTEMPTS"
_GSB_LABELS = {"A>>B", "A>B", "A=B", "A<B", "A<<B"}
_LOCAL_EXCEL_HELPER_NAMES = {
    "excel_profile_sheet",
    "read_data_from_excel_compact",
    "excel_filter_rows",
    "excel_aggregate",
}


@dataclass(frozen=True)
class _DirectionRunResult:
    direction: str
    payload: dict[str, Any]
    raw_response: str
    attempts: list[dict[str, Any]]
    failure_reason: str = ""


def agent_context(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    instance: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
):
    """Build a run-scoped GSB judge agent with optional MCP tools."""

    return gdpval_judge_agent_context(
        provider=provider,
        cwd=Path(cwd),
        request_extra=request_extra,
        instance=instance,
        context=context,
        name="gdpval_gsb_judge",
        role="Compare GDPVal deliverables and write a GSB JSON verdict.",
        system_prompt=GDPVAL_GSB_JUDGE_SYSTEM_PROMPT,
        include_excel_helpers=True,
        timeout_seconds=timeout_seconds,
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    workdir_path = Path(workdir)
    input_dir = _input_dir_for(workdir_path)
    rubrics = normalize_rubrics(instance.get("rubrics"))
    result_path = workdir_path / JUDGE_RESULT_FILE
    candidate_result = instance.get("candidate_result") or {}
    return "\n".join(
        [
            "Judge this GDPVal candidate submission with GSB comparison.",
            "",
            "## Paths",
            f"- WORKDIR: {workdir_path}",
            f"- CANDIDATE_DIR: {input_dir / 'candidate'}",
            f"- GOLD_DIR: {input_dir / 'gold'}",
            f"- REFERENCE_DIR: {input_dir / 'reference'}",
            f"- ZIP_EXTRACTS: {input_dir / '__zip_extracts'}",
            f"- REQUIRED_OUTPUT_JSON: {result_path}",
            "",
            "## Direction Definitions",
            "- reverse: A is GOLD_DIR, B is CANDIDATE_DIR.",
            "- forward: A is CANDIDATE_DIR, B is GOLD_DIR.",
            "",
            "## Original Task Prompt",
            str(instance.get("prompt") or ""),
            "",
            "## Candidate Result Summary",
            json.dumps(
                _candidate_summary(candidate_result), ensure_ascii=False, indent=2
            ),
            "",
            "## Rubrics",
            json.dumps(rubrics, ensure_ascii=False, indent=2),
            "",
            "## Instructions",
            "- The suite will run reverse and forward as isolated judge prompts.",
            "- Each direction must inspect the supplied files with tools before "
            "scoring when files are available.",
            "- The harness will aggregate parsed direction outputs to "
            "REQUIRED_OUTPUT_JSON.",
        ]
    )


def apply_oracle(workspace: Path, instance: Mapping[str, Any]) -> None:
    """Model-free smoke path: write a deterministic tie against gold."""

    rubrics = normalize_rubrics(instance.get("rubrics"))
    reverse = _oracle_direction(
        rubrics,
        grade_a=1.0,
        grade_b=1.0,
        gsb="A=B",
        final_gsb="A=B",
        explanation="oracle judge marks candidate and gold as tied",
    )
    payload = {"reverse": reverse, "forward": reverse}
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / JUDGE_RESULT_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
    task: str,
    max_turns: int,
    timeout_seconds: float | None = None,
) -> tuple[State, Any]:
    """Run swalm-style independent reverse/forward GSB judge directions."""

    workdir = Path(cwd)
    state = State(task=task)

    def events():
        if not _has_available_files(context.get("gold_manifest")):
            return

        paths = _judge_paths(context)
        if not paths["candidate"]:
            return

        rubrics = normalize_rubrics(instance.get("rubrics"))
        max_direction_attempts = _max_gsb_direction_attempts(instance)
        final_answer_summary = json.dumps(
            _candidate_summary(instance.get("candidate_result") or {}),
            ensure_ascii=False,
            indent=2,
        )
        all_attempts: list[dict[str, Any]] = []

        reverse = yield from _run_direction(
            provider=provider,
            cwd=workdir,
            request_extra=request_extra,
            instance=instance,
            context=context,
            combined_state=state,
            direction="reverse",
            a_label="standard answer",
            b_label="candidate",
            a_paths=paths["gold"],
            b_paths=paths["candidate"],
            reference_paths=paths["reference"],
            zip_paths=paths["zip"],
            final_answer_label="B Final Answer Summary",
            final_answer_summary=final_answer_summary,
            rubrics=rubrics,
            require_tool=bool(paths["candidate"]),
            max_turns=max_turns,
            max_attempts=max_direction_attempts,
            timeout_seconds=timeout_seconds,
        )
        all_attempts.extend(reverse.attempts)

        forward: _DirectionRunResult | None = None
        if paths["candidate"] and not reverse.failure_reason:
            forward = yield from _run_direction(
                provider=provider,
                cwd=workdir,
                request_extra=request_extra,
                instance=instance,
                context=context,
                combined_state=state,
                direction="forward",
                a_label="candidate",
                b_label="standard answer",
                a_paths=paths["candidate"],
                b_paths=paths["gold"],
                reference_paths=paths["reference"],
                zip_paths=paths["zip"],
                final_answer_label="A Final Answer Summary",
                final_answer_summary=final_answer_summary,
                rubrics=rubrics,
                require_tool=True,
                max_turns=max_turns,
                max_attempts=max_direction_attempts,
                timeout_seconds=timeout_seconds,
            )
            all_attempts.extend(forward.attempts)

        payload: dict[str, Any] = {
            "reverse": reverse.payload,
            "forward": forward.payload if forward is not None else {},
            "_sal_judge_attempt_summary": _attempt_summary(all_attempts),
        }
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / JUDGE_RESULT_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (workdir / JUDGE_ATTEMPTS_FILE).write_text(
            json.dumps({"attempts": all_attempts}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return state, events()


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse the GSB judge JSON file and compute GDPVal GSB scores."""

    workdir = Path(workspace)
    result_file = workdir / JUDGE_RESULT_FILE
    rubrics = normalize_rubrics(instance.get("rubrics"))
    max_score = sum(item["weight"] for item in rubrics)
    base: dict[str, Any] = {
        "task_id": str(instance.get("task_id") or instance.get("instance_id") or ""),
        "judge_mode": "gsb",
        "judge_result_file": str(result_file),
    }
    paths: dict[str, list[str]] = {"candidate": [], "gold": []}
    if context:
        paths = _judge_paths(context)
        base["input_dir"] = str(context.get("input_dir") or "")
        base["candidate_dir"] = str(context.get("candidate_dir") or "")
        base["gold_dir"] = str(context.get("gold_dir") or "")
        base["reference_dir"] = str(context.get("reference_dir") or "")
        base["gold_manifest"] = context.get("gold_manifest") or []
        if not _has_available_files(context.get("gold_manifest")):
            return {
                **base,
                "status": "gold_deliverables_missing",
                "score": 0.0,
                "earned_score": 0.0,
                "max_score": max_score,
                "overall_explanation_reverse": (
                    "no readable standard-answer deliverable files were staged"
                ),
                "overall_explanation_forward": (
                    "no readable standard-answer deliverable files were staged"
                ),
            }
        if not paths["candidate"]:
            return {
                **base,
                "status": "candidate_deliverables_missing",
                "score": 0.0,
                "earned_score": 0.0,
                "max_score": max_score,
                "overall_explanation_reverse": (
                    "no readable candidate deliverable files were staged"
                ),
                "overall_explanation_forward": (
                    "no readable candidate deliverable files were staged"
                ),
            }

    if not result_file.is_file():
        return {
            **base,
            "status": "judge_result_missing",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": max_score,
            "overall_explanation_reverse": (
                "judge did not write the required GSB JSON file"
            ),
            "overall_explanation_forward": (
                "judge did not write the required GSB JSON file"
            ),
        }
    raw = result_file.read_text(encoding="utf-8", errors="replace")
    try:
        payload = parse_gsb_judge_payload(raw)
        failure_reason = _payload_failure_reason(
            payload,
            require_forward=bool(paths["candidate"]),
            require_rubrics=bool(rubrics),
        )
        if failure_reason:
            return {
                **base,
                "status": "judge_result_invalid",
                "score": 0.0,
                "earned_score": 0.0,
                "max_score": max_score,
                "overall_explanation_reverse": failure_reason,
                "overall_explanation_forward": failure_reason,
                "raw_judge_result": raw[:20_000],
                **_attempt_result_fields(workdir),
            }
        scored = score_gsb_judgment(payload, instance.get("rubrics"))
    except ValueError as exc:
        return {
            **base,
            "status": "judge_result_invalid",
            "score": 0.0,
            "earned_score": 0.0,
            "max_score": max_score,
            "overall_explanation_reverse": f"{type(exc).__name__}: {exc}",
            "overall_explanation_forward": f"{type(exc).__name__}: {exc}",
            "raw_judge_result": raw[:20_000],
            **_attempt_result_fields(workdir),
        }
    return {
        **base,
        **scored,
        "raw_judge_result_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        **_attempt_result_fields(workdir),
    }


def _run_direction(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
    combined_state: State,
    direction: str,
    a_label: str,
    b_label: str,
    a_paths: list[str],
    b_paths: list[str],
    reference_paths: list[str],
    zip_paths: list[str],
    final_answer_label: str,
    final_answer_summary: str,
    rubrics: list[dict[str, Any]],
    require_tool: bool,
    max_turns: int,
    max_attempts: int,
    timeout_seconds: float | None,
):
    attempts: list[dict[str, Any]] = []
    warning: str | None = None
    last_response = ""
    last_payload: dict[str, Any] = {}
    failure_reason = ""
    for attempt_index in range(max_attempts):
        prompt = _build_direction_prompt(
            instance=instance,
            direction=direction,
            a_label=a_label,
            b_label=b_label,
            a_paths=a_paths,
            b_paths=b_paths,
            reference_paths=reference_paths,
            zip_paths=zip_paths,
            final_answer_label=final_answer_label,
            final_answer_summary=final_answer_summary,
            rubrics=rubrics,
            warning=warning,
        )
        with gdpval_judge_agent_context(
            provider=provider,
            cwd=cwd,
            request_extra=request_extra,
            instance=instance,
            context=context,
            name=f"gdpval_gsb_judge_{direction}_{attempt_index + 1}",
            role="Compare GDPVal deliverables and emit one GSB direction.",
            system_prompt=GDPVAL_GSB_JUDGE_SYSTEM_PROMPT,
            include_excel_helpers=True,
            path_label_roles=_path_label_roles_for_direction(direction),
            timeout_seconds=timeout_seconds,
        ) as agent:
            source_state, source_events = agent.run(prompt, max_turns=max_turns)
            copied = yield from _copy_new_events(source_state, combined_state, start=0)
            for _ in source_events:
                copied = yield from _copy_new_events(
                    source_state, combined_state, start=copied
                )

        last_response = _last_assistant_text(source_state)
        tool_names = _tool_names(source_state)
        payload, parse_error = _parse_direction_payload_safely(last_response)
        last_payload = payload
        final_gsb = str((payload.get("overall") or {}).get("final_gsb") or "").strip()
        has_rubrics = bool(payload.get("rubrics_result"))
        parseable = bool(payload)
        valid_final_gsb = final_gsb in _GSB_LABELS
        valid_output = parseable and valid_final_gsb and (has_rubrics or not rubrics)

        reason = ""
        if require_tool and not tool_names:
            reason = "no_tool_messages"
            warning = "no_tool"
        elif not valid_output:
            reason = parse_error or "invalid_output"
            warning = "invalid_output"
        else:
            record = _attempt_record(
                direction=direction,
                attempt_index=attempt_index,
                tool_names=tool_names,
                parseable=parseable,
                valid_final_gsb=valid_final_gsb,
                failure_reason="",
                raw_response=last_response,
            )
            attempts.append(record)
            return _DirectionRunResult(
                direction=direction,
                payload=payload,
                raw_response=last_response,
                attempts=attempts,
            )

        attempts.append(
            _attempt_record(
                direction=direction,
                attempt_index=attempt_index,
                tool_names=tool_names,
                parseable=parseable,
                valid_final_gsb=valid_final_gsb,
                failure_reason=reason,
                raw_response=last_response,
            )
        )
        failure_reason = reason

    if failure_reason == "no_tool_messages":
        last_payload = {}
        last_response = (
            f"{direction} judge failed to invoke any tool after maximum retries; "
            "treating evaluation as failed with score 0.0."
        )
    return _DirectionRunResult(
        direction=direction,
        payload=last_payload,
        raw_response=last_response,
        attempts=attempts,
        failure_reason=failure_reason,
    )


def _parse_direction_payload_safely(raw_response: str) -> tuple[dict[str, Any], str]:
    try:
        return parse_gsb_direction_payload(raw_response), ""
    except ValueError as exc:
        return {}, f"invalid_output: {type(exc).__name__}: {exc}"


def _path_label_roles_for_direction(direction: str) -> dict[str, str]:
    if direction == "forward":
        return {"A": "candidate", "B": "gold"}
    return {"A": "gold", "B": "candidate"}


def _payload_failure_reason(
    payload: Mapping[str, Any],
    *,
    require_forward: bool = True,
    require_rubrics: bool = False,
) -> str:
    summary = payload.get("_sal_judge_attempt_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    for direction in ("reverse", "forward"):
        item = summary.get(direction)
        if not isinstance(item, Mapping):
            continue
        reason = str(item.get("last_failure_reason") or "").strip()
        if reason:
            return f"{direction} judge failed: {reason}"
    for direction in ("reverse", "forward"):
        if direction == "forward" and not require_forward:
            continue
        reason = _direction_payload_failure_reason(
            payload.get(direction),
            direction=direction,
            require_rubrics=require_rubrics,
        )
        if reason:
            return reason
    return ""


def _direction_payload_failure_reason(
    payload: Any,
    *,
    direction: str,
    require_rubrics: bool,
) -> str:
    if not isinstance(payload, Mapping) or not payload:
        return f"{direction} judge payload is missing"
    raw_rubrics = payload.get("rubrics_result") or payload.get("rubric_results")
    if require_rubrics and not _has_rubric_results(raw_rubrics):
        return f"{direction} judge rubrics_result is missing"
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        overall = payload
    final_gsb = str((overall or {}).get("final_gsb") or "").strip()
    if final_gsb not in _GSB_LABELS:
        return f"{direction} judge final_gsb is missing or invalid"
    return ""


def _has_rubric_results(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return any(isinstance(item, Mapping) for item in value)


def _copy_new_events(
    source_state: State,
    combined_state: State,
    *,
    start: int,
):
    for event in source_state.events[start:]:
        yield combined_state.record_event(event)
    return len(source_state.events)


def _build_direction_prompt(
    *,
    instance: Mapping[str, Any],
    direction: str,
    a_label: str,
    b_label: str,
    a_paths: list[str],
    b_paths: list[str],
    reference_paths: list[str],
    zip_paths: list[str],
    final_answer_label: str,
    final_answer_summary: str,
    rubrics: list[dict[str, Any]],
    warning: str | None,
) -> str:
    warning_lines: list[str] = []
    if warning == "no_tool":
        warning_lines.extend(
            [
                "- WARNING: In the previous round you did NOT invoke any document "
                "or filesystem tool. This violates the requirements.",
                "- In this round, before producing any score, you MUST actually "
                "invoke tools to read relevant supplied files from both the <A> "
                "side and the <B> side.",
            ]
        )
    elif warning == "invalid_output":
        warning_lines.extend(
            [
                "- WARNING: In the previous round your response did not output a "
                "valid <rubrics_result> block and/or <overall> block with a valid "
                "final_gsb value.",
                "- Please try again and strictly follow the required output format.",
            ]
        )
    return "\n".join(
        [
            f"<judge_direction>{direction}</judge_direction>",
            "",
            "<prompt>",
            str(instance.get("prompt") or ""),
            "</prompt>",
            "",
            "<rubrics>",
            json.dumps(_binary_judge_rubrics(rubrics), ensure_ascii=False, indent=2),
            "</rubrics>",
            "",
            "<reference_file_urls>",
            "The following reference/input files were provided as task inputs "
            "(use tools to read as needed, absolute paths):",
            _path_block(reference_paths),
            "</reference_file_urls>",
            "",
            "<A>",
            f"A files ({a_label}; use tools to read, absolute paths):",
            _path_block(a_paths),
            f"{final_answer_label}:" if final_answer_label.startswith("A ") else "",
            final_answer_summary if final_answer_label.startswith("A ") else "",
            "</A>",
            "",
            "<B>",
            f"B files ({b_label}; use tools to read, absolute paths):",
            _path_block(b_paths),
            f"{final_answer_label}:" if final_answer_label.startswith("B ") else "",
            final_answer_summary if final_answer_label.startswith("B ") else "",
            "</B>",
            "",
            "Zip archives automatically extracted for judging:",
            _path_block(zip_paths),
            "",
            "#### Tool Usage Requirements (Important)",
            *warning_lines,
            "- You MUST use MCP document/filesystem tools or the local judge "
            "inspection tools to actually open and read relevant files from both "
            "the <A> side and the <B> side before scoring when files are listed. "
            "Do NOT rely on file paths or the Final Answer Summary alone.",
            "- Prefer targeted inspection of the files needed by the rubrics. Do "
            "not read every file in a large directory, archive, workbook, or "
            "notebook once the decisive evidence is already available.",
            "- If no tool is invoked in the current round when files are listed, "
            "the scoring process will be marked as failed.",
            "",
            GDPVAL_GSB_JUDGE_EXCEL_HANDLING_PROMPT,
            "",
            "#### Required Output Format",
            "Wrap a JSON array in <rubrics_result> and </rubrics_result>. Each "
            "item must contain exactly: score, criterion, grade_A, grade_B, gsb, "
            "grade_explanation.",
            'Then output <overall>{"overall_explanation": "...", '
            '"final_gsb": "A=B"}</overall>.',
        ]
    )


def _judge_paths(context: Mapping[str, Any]) -> dict[str, list[str]]:
    candidate = _manifest_paths(context.get("candidate_manifest"))
    gold = _manifest_paths(context.get("gold_manifest"))
    reference = _manifest_paths(context.get("reference_manifest"))
    zip_paths = _manifest_paths(context.get("zip_manifest"))
    candidate.extend(_zip_paths_for_label(context.get("zip_manifest"), "candidate"))
    gold.extend(_zip_paths_for_label(context.get("zip_manifest"), "gold"))
    reference.extend(_zip_paths_for_label(context.get("zip_manifest"), "reference"))
    return {
        "candidate": sorted(set(candidate)),
        "gold": sorted(set(gold)),
        "reference": sorted(set(reference)),
        "zip": sorted(set(zip_paths)),
    }


def _manifest_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        status = str(value.get("status") or "")
        if not value.get("missing") and status not in {"skipped_unsafe", "missing"}:
            path = value.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
        for key in ("entries", "files"):
            paths.extend(_manifest_paths(value.get(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            paths.extend(_manifest_paths(item))
    return paths


def _zip_paths_for_label(value: Any, label: str) -> list[str]:
    marker = f"/__zip_extracts/{label}/"
    return [path for path in _manifest_paths(value) if marker in path]


def _binary_judge_rubrics(rubrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"score": item["weight"], "criterion": item["criterion"]} for item in rubrics
    ]


def _path_block(paths: list[str]) -> str:
    return "\n".join(f"- {path}" for path in paths) if paths else "(none)"


def _last_assistant_text(state: State) -> str:
    for message in reversed(state.messages):
        if message.sender.startswith("gdpval_gsb_judge"):
            text = text_of(message.content).strip()
            if text:
                return text
    return ""


def _tool_names(state: State) -> list[str]:
    return [
        event.tool_name
        for event in state.events
        if isinstance(event, ToolExecutionStartEvent)
    ]


def _attempt_record(
    *,
    direction: str,
    attempt_index: int,
    tool_names: list[str],
    parseable: bool,
    valid_final_gsb: bool,
    failure_reason: str,
    raw_response: str,
) -> dict[str, Any]:
    mcp_tools = [name for name in tool_names if _is_mcp_tool_name(name)]
    return {
        "direction": direction,
        "attempt": attempt_index + 1,
        "has_tool_messages": bool(tool_names),
        "tool_message_count": len(tool_names),
        "tool_names": tool_names,
        "mcp_tool_count": len(mcp_tools),
        "mcp_tool_names": mcp_tools,
        "parseable": parseable,
        "valid_final_gsb": valid_final_gsb,
        "failure_reason": failure_reason,
        "raw_response_preview": raw_response[:4000],
    }


def _is_mcp_tool_name(name: str) -> bool:
    if name in _LOCAL_EXCEL_HELPER_NAMES:
        return False
    return name.startswith(("filesystem_", "pdf_", "excel_", "word_", "ppt_"))


def _attempt_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        direction = str(attempt.get("direction") or "")
        item = by_direction.setdefault(
            direction,
            {
                "attempts": 0,
                "had_tool_messages": False,
                "mcp_tool_count": 0,
                "last_failure_reason": "",
            },
        )
        item["attempts"] += 1
        item["had_tool_messages"] = bool(
            item["had_tool_messages"] or attempt.get("has_tool_messages")
        )
        item["mcp_tool_count"] += int(attempt.get("mcp_tool_count") or 0)
        item["last_failure_reason"] = str(attempt.get("failure_reason") or "")
    return by_direction


def _attempt_result_fields(workdir: Path) -> dict[str, Any]:
    path = workdir / JUDGE_ATTEMPTS_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"judge_attempts_file": str(path)}
    attempts = payload.get("attempts") if isinstance(payload, Mapping) else []
    if not isinstance(attempts, list):
        attempts = []
    return {
        "judge_attempts_file": str(path),
        "judge_attempts": attempts,
        "judge_retry_summary": _attempt_summary(
            [item for item in attempts if isinstance(item, Mapping)]
        ),
    }


def _max_gsb_direction_attempts(instance: Mapping[str, Any]) -> int:
    value = instance.get("judge_gsb_direction_attempts")
    if value is None:
        value = os.environ.get(GSB_DIRECTION_ATTEMPTS_ENV)
    return _coerce_gsb_direction_attempts(value)


def _coerce_gsb_direction_attempts(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_GSB_DIRECTION_ATTEMPTS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_GSB_DIRECTION_ATTEMPTS
    return max(1, min(MAX_GSB_DIRECTION_ATTEMPTS, parsed))


def _oracle_direction(
    rubrics: list[dict[str, Any]],
    *,
    grade_a: float,
    grade_b: float,
    gsb: str,
    final_gsb: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "rubrics_result": [
            {
                "index": item["index"],
                "score": item["weight"],
                "criterion": item["criterion"],
                "grade_A": grade_a,
                "grade_B": grade_b,
                "gsb": gsb,
                "grade_explanation": explanation,
            }
            for item in rubrics
        ],
        "overall": {
            "overall_explanation": explanation,
            "final_gsb": final_gsb,
        },
    }


def _has_available_files(manifest: Any) -> bool:
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        return False
    for item in manifest:
        if not isinstance(item, Mapping):
            continue
        if not item.get("missing") and item.get("path"):
            return True
    return False
