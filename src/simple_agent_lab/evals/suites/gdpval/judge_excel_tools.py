"""Local Excel helper tools for GDPVal judge runs.

These mirror the swalm judge's Excel helper surface without importing swalm.
They are intentionally optional at runtime: if ``openpyxl`` is unavailable in
the eval image, each tool returns a JSON error that the judge can recover from
by using the MCP Excel tools or local bash/Python.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, cast

from simple_agent_lab.tools import AgentTool, ToolResult, text_result


def _openpyxl_attr(module_name: str, attr: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency
        raise RuntimeError(f"openpyxl_import_failed: {exc}") from exc
    return getattr(module, attr)


def _load_workbook() -> Any:
    return _openpyxl_attr("openpyxl", "load_workbook")


def _get_column_letter() -> Any:
    return _openpyxl_attr("openpyxl.utils", "get_column_letter")


def _coordinate_to_tuple() -> Any:
    return _openpyxl_attr("openpyxl.utils.cell", "coordinate_to_tuple")


def _column_index_from_string() -> Any:
    return _openpyxl_attr("openpyxl.utils.cell", "column_index_from_string")


def make_judge_excel_tools(
    *, workdir: str | Path, reference_dir: str | Path
) -> tuple[AgentTool, ...]:
    """Return compact, read-only Excel inspection tools for the judge."""

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()

    def excel_profile_sheet(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, abort, on_update
        return _excel_action_result(
            "profile", args, workspace=workspace, references=references
        )

    def read_data_from_excel_compact(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, abort, on_update
        return _excel_action_result(
            "compact_read", args, workspace=workspace, references=references
        )

    def excel_filter_rows(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, abort, on_update
        return _excel_action_result(
            "filter_rows", args, workspace=workspace, references=references
        )

    def excel_aggregate(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, abort, on_update
        return _excel_action_result(
            "aggregate", args, workspace=workspace, references=references
        )

    return (
        AgentTool(
            name="excel_profile_sheet",
            description=(
                "Profile an Excel worksheet without dumping all cells. Use this "
                "first for large or unknown sheets to get dimensions, non-empty "
                "range, headers, and sample non-empty rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the Excel workbook.",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Worksheet name. Defaults to the first sheet.",
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Number of non-empty sample rows to return.",
                        "default": 5,
                    },
                    "max_scan_rows": {
                        "type": "integer",
                        "description": "Maximum rows to scan for the non-empty range.",
                        "default": 20000,
                    },
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
            execute=excel_profile_sheet,
        ),
        AgentTool(
            name="read_data_from_excel_compact",
            description=(
                "Read a bounded Excel range as compact row arrays. Prefer this "
                "over full-sheet reads because it omits per-cell metadata and "
                "caps output rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "start_cell": {"type": "string", "default": "A1"},
                    "end_cell": {"type": "string"},
                    "max_rows": {"type": "integer", "default": 200},
                    "include_empty_rows": {"type": "boolean", "default": False},
                    "formulas": {"type": "boolean", "default": False},
                    "max_scan_rows": {"type": "integer", "default": 20000},
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
            execute=read_data_from_excel_compact,
        ),
        AgentTool(
            name="excel_filter_rows",
            description=(
                "Filter an Excel sheet by header-based conditions and return "
                "matching rows compactly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "header_row": {"type": "integer", "default": 1},
                    "filters": {
                        "type": "object",
                        "description": (
                            "Column filters keyed by header/letter. Values may be "
                            "exact values or objects with contains, equals, gt, "
                            "ge, lt, le, or non_empty."
                        ),
                        "default": {},
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "max_rows": {"type": "integer", "default": 200},
                    "max_scan_rows": {"type": "integer", "default": 20000},
                    "case_sensitive": {"type": "boolean", "default": False},
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
            execute=excel_filter_rows,
        ),
        AgentTool(
            name="excel_aggregate",
            description=(
                "Compute simple aggregates over an Excel sheet, optionally "
                "grouped and filtered. Use it for sums, counts, means, min, "
                "and max checks instead of dumping raw rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "header_row": {"type": "integer", "default": 1},
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "object"},
                        "default": [{"op": "count", "name": "count"}],
                    },
                    "filters": {"type": "object", "default": {}},
                    "max_scan_rows": {"type": "integer", "default": 10000},
                    "case_sensitive": {"type": "boolean", "default": False},
                },
                "required": ["filepath"],
                "additionalProperties": False,
            },
            execute=excel_aggregate,
        ),
    )


def _excel_action_result(
    action: str,
    args: Mapping[str, Any],
    *,
    workspace: Path,
    references: Path,
) -> ToolResult:
    try:
        payload = _run_excel_action(
            action, args, workspace=workspace, references=references
        )
        return text_result(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:  # noqa: BLE001 - model-visible diagnostic
        return text_result(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "action": action,
                    "diagnostics": _excel_error_diagnostics(
                        args, workspace=workspace, references=references
                    ),
                },
                ensure_ascii=False,
                default=str,
            ),
            is_error=True,
        )


def _run_excel_action(
    action: str,
    args: Mapping[str, Any],
    *,
    workspace: Path,
    references: Path,
) -> dict[str, Any]:
    if action == "profile":
        return _action_profile(args, workspace=workspace, references=references)
    if action == "compact_read":
        return _action_compact_read(args, workspace=workspace, references=references)
    if action == "filter_rows":
        return _action_filter_rows(args, workspace=workspace, references=references)
    if action == "aggregate":
        return _action_aggregate(args, workspace=workspace, references=references)
    raise ValueError(f"unknown Excel action {action!r}")


def _load_sheet(
    args: Mapping[str, Any],
    *,
    workspace: Path,
    references: Path,
    data_only: bool,
):
    load_workbook = _load_workbook()
    path = _resolve_workbook_path(args, workspace=workspace, references=references)
    workbook = load_workbook(filename=path, read_only=True, data_only=data_only)
    sheet_name = _choose_sheet_name(
        workbook, args.get("sheet_name") or args.get("sheet")
    )
    return path, workbook, workbook[sheet_name], sheet_name


def _action_profile(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> dict[str, Any]:
    get_column_letter = _get_column_letter()
    path, workbook, sheet, sheet_name = _load_sheet(
        args, workspace=workspace, references=references, data_only=False
    )
    sample_rows = max(0, int(args.get("sample_rows", 5) or 5))
    bbox = _sheet_bbox(sheet, max_scan_rows=int(args.get("max_scan_rows", 20000)))
    samples: list[dict[str, Any]] = []
    if bbox["min_row"] is not None:
        for row_idx, row in enumerate(
            sheet.iter_rows(
                min_row=bbox["min_row"],
                max_row=bbox["max_row"],
                min_col=bbox["min_col"],
                max_col=bbox["max_col"],
                values_only=True,
            ),
            start=bbox["min_row"],
        ):
            values = [_clean_value(value) for value in row]
            if any(value is not None and str(value).strip() for value in values):
                samples.append({"row": row_idx, "values": values})
                if len(samples) >= sample_rows:
                    break

    max_row = _safe_max_row(sheet)
    max_col = _safe_max_col(sheet)
    nominal_cells = max(1, max_row * max_col)
    blank_ratio = 1.0 - min(1.0, bbox["non_empty_cells"] / nominal_cells)
    return {
        "filepath": str(path),
        "sheet_name": sheet_name,
        "available_sheets": workbook.sheetnames,
        "reported_dimension": f"A1:{get_column_letter(max_col)}{max_row}",
        "reported_max_row": max_row,
        "reported_max_column": max_col,
        "non_empty_bbox": bbox,
        "blank_ratio_vs_reported_dimension": blank_ratio,
        "sample_non_empty_rows": samples,
    }


def _action_compact_read(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> dict[str, Any]:
    get_column_letter = _get_column_letter()
    coordinate_to_tuple = _coordinate_to_tuple()
    formulas = bool(args.get("formulas", False))
    path, _workbook, sheet, sheet_name = _load_sheet(
        args, workspace=workspace, references=references, data_only=not formulas
    )
    start_cell = str(args.get("start_cell") or args.get("start") or "A1")
    start_row, start_col = coordinate_to_tuple(start_cell)
    max_rows = max(1, min(int(args.get("max_rows", 200) or 200), 1000))
    include_empty_rows = bool(args.get("include_empty_rows", False))

    if args.get("end_cell") or args.get("end"):
        end_row, end_col = coordinate_to_tuple(str(args.get("end_cell") or args["end"]))
    else:
        bbox = _sheet_bbox(
            sheet, max_scan_rows=int(args.get("max_scan_rows", 20000) or 20000)
        )
        end_row = int(bbox["max_row"] or start_row)
        end_col = int(bbox["max_col"] or max(start_col, _safe_max_col(sheet)))
    requested_end_row = end_row
    end_row = min(end_row, start_row + max_rows - 1)
    end_col = max(start_col, end_col)

    rows: list[list[Any]] = []
    row_numbers: list[int] = []
    non_empty_cells = 0
    for row_idx, row in enumerate(
        sheet.iter_rows(
            min_row=start_row,
            max_row=end_row,
            min_col=start_col,
            max_col=end_col,
            values_only=True,
        ),
        start=start_row,
    ):
        values = [_clean_value(value) for value in row]
        row_non_empty = sum(
            1 for value in values if value is not None and str(value).strip()
        )
        non_empty_cells += row_non_empty
        if row_non_empty or include_empty_rows:
            row_numbers.append(row_idx)
            rows.append(values)

    return {
        "filepath": str(path),
        "sheet_name": sheet_name,
        "range": f"{get_column_letter(start_col)}{start_row}:"
        f"{get_column_letter(end_col)}{end_row}",
        "requested_end_row": requested_end_row,
        "returned_rows": len(rows),
        "returned_cols": end_col - start_col + 1,
        "non_empty_cells": non_empty_cells,
        "truncated": requested_end_row > end_row,
        "row_numbers": row_numbers,
        "rows": rows,
    }


def _action_filter_rows(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> dict[str, Any]:
    get_column_letter = _get_column_letter()
    path, _workbook, sheet, sheet_name = _load_sheet(
        args, workspace=workspace, references=references, data_only=True
    )
    header_row = max(1, int(args.get("header_row", 1) or 1))
    raw_filters = args.get("filters")
    filters: Mapping[str, Any] = (
        cast("Mapping[str, Any]", raw_filters)
        if isinstance(raw_filters, Mapping)
        else {}
    )
    columns = args.get("columns") if isinstance(args.get("columns"), Sequence) else []
    if isinstance(columns, (str, bytes)):
        columns = [str(columns)]
    max_rows = max(1, min(int(args.get("max_rows", 200) or 200), 1000))
    max_scan_rows = max(1, int(args.get("max_scan_rows", 20000) or 20000))
    case_sensitive = bool(args.get("case_sensitive", False))

    requested_columns = _collect_requested_columns(args)
    original_header_row = header_row
    header_row, headers, header_auto_correction = _apply_header_auto_correction(
        sheet,
        header_row,
        requested_columns,
    )
    if columns:
        selected_indices = [_resolve_column(column, headers) for column in columns]
    else:
        selected_indices = list(range(1, len(headers) + 1))

    matched: list[OrderedDict[str, Any]] = []
    scanned = 0
    total_matched = 0
    for row_idx, row in enumerate(
        sheet.iter_rows(min_row=header_row + 1, values_only=True),
        start=header_row + 1,
    ):
        if scanned >= max_scan_rows:
            break
        scanned += 1
        row_values = [_clean_value(value) for value in row]
        if not _row_matches(row_values, filters, headers, case_sensitive):
            continue
        total_matched += 1
        if len(matched) < max_rows:
            item: OrderedDict[str, Any] = OrderedDict()
            item["_row"] = row_idx
            for col_idx in selected_indices:
                header = (
                    headers[col_idx - 1]
                    if col_idx - 1 < len(headers)
                    else get_column_letter(col_idx)
                )
                item[str(header)] = (
                    row_values[col_idx - 1] if col_idx - 1 < len(row_values) else None
                )
            matched.append(item)

    return {
        "filepath": str(path),
        "sheet_name": sheet_name,
        "header_row": header_row,
        "requested_header_row": original_header_row,
        "header_auto_correction": header_auto_correction,
        "headers": headers,
        "filters": dict(filters),
        "scanned_rows": scanned,
        "total_matched": total_matched,
        "returned_rows": len(matched),
        "truncated": total_matched > len(matched),
        "rows": matched,
    }


def _action_aggregate(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> dict[str, Any]:
    path, _workbook, sheet, sheet_name = _load_sheet(
        args, workspace=workspace, references=references, data_only=True
    )
    header_row = max(1, int(args.get("header_row", 1) or 1))
    raw_filters = args.get("filters")
    filters: Mapping[str, Any] = (
        cast("Mapping[str, Any]", raw_filters)
        if isinstance(raw_filters, Mapping)
        else {}
    )
    group_by = _as_list(args.get("group_by"))
    metrics = args.get("metrics") or [{"op": "count", "name": "count"}]
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        metrics = [{"column": metrics, "op": "sum"}]
    max_scan_rows = max(1, int(args.get("max_scan_rows", 10000) or 10000))
    case_sensitive = bool(args.get("case_sensitive", False))

    requested_columns = _collect_requested_columns(args, include_return_columns=False)
    original_header_row = header_row
    header_row, headers, header_auto_correction = _apply_header_auto_correction(
        sheet,
        header_row,
        requested_columns,
    )
    group_indices = [_resolve_column(column, headers) for column in group_by]
    metric_specs = [_metric_spec(metric, headers) for metric in metrics]

    groups: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    scanned = 0
    matched = 0
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        if scanned >= max_scan_rows:
            break
        scanned += 1
        row_values = [_clean_value(value) for value in row]
        if not _row_matches(row_values, filters, headers, case_sensitive):
            continue
        matched += 1
        key: tuple[Any, ...] = tuple(
            row_values[idx - 1] if idx - 1 < len(row_values) else None
            for idx in group_indices
        )
        if not key:
            key = cast("tuple[Any, ...]", ("__all__",))
        if key not in groups:
            groups[key] = {
                "_rows": 0,
                "_metrics": {
                    spec["name"]: {"count": 0, "sum": 0.0, "min": None, "max": None}
                    for spec in metric_specs
                },
            }
        groups[key]["_rows"] += 1
        for spec in metric_specs:
            _update_metric(groups[key]["_metrics"][spec["name"]], spec, row_values)

    results: list[OrderedDict[str, Any]] = []
    for key, state in groups.items():
        item: OrderedDict[str, Any] = OrderedDict()
        for name, value in zip(group_by, key):
            item[str(name)] = value
        item["row_count"] = state["_rows"]
        for spec in metric_specs:
            item[spec["name"]] = _finish_metric(state["_metrics"][spec["name"]], spec)
        results.append(item)

    return {
        "filepath": str(path),
        "sheet_name": sheet_name,
        "header_row": header_row,
        "requested_header_row": original_header_row,
        "header_auto_correction": header_auto_correction,
        "headers": headers,
        "group_by": group_by,
        "filters": dict(filters),
        "metrics": list(metrics),
        "scanned_rows": scanned,
        "matched_rows": matched,
        "scan_truncated": scanned >= max_scan_rows,
        "groups": results,
    }


def _resolve_workbook_path(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> Path:
    value = (
        args.get("filepath")
        or args.get("file_path")
        or args.get("filename")
        or args.get("path")
    )
    if not value:
        raise ValueError("Missing required filepath")
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
    label_prefix = ""
    rel_text = text
    if re.match(r"^[AB][\\/]", text):
        label_prefix = text[0].upper()
        rel_text = text[2:]
    rel_path = Path(rel_text)

    candidates = []
    if not path.is_absolute():
        if not label_prefix:
            candidates.extend((workspace / path, references / path))
        candidates.extend((workspace / rel_path, references / rel_path))
    else:
        candidates.append(path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    basename = rel_path.name or path.name
    if not basename:
        return path.resolve()
    roots = [
        workspace,
        references,
        workspace.parent,
        Path("/app/workspace/gdpevals"),
        Path("/app/workspace"),
        Path("/workspace"),
    ]
    seen: set[Path] = set()
    matches: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob(basename):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                matches.append(candidate.resolve())
                if len(matches) >= 50:
                    break
    if matches:
        return sorted(
            matches,
            key=lambda item: _workbook_path_priority(item, label_prefix=label_prefix),
        )[0]
    raise FileNotFoundError(text)


def _choose_sheet_name(workbook: Any, requested: Any) -> str:
    if not requested:
        return workbook.sheetnames[0]
    text = str(requested).strip()
    if text in workbook.sheetnames:
        return text
    lowered = text.lower()
    for name in workbook.sheetnames:
        if name.lower() == lowered:
            return name
    normalized = _normalize_name(text)
    for name in workbook.sheetnames:
        if _normalize_name(name) == normalized:
            return name
    tokenized = _normalize_name_tokens(text)
    for name in workbook.sheetnames:
        if _normalize_name_tokens(name) == tokenized:
            return name
    if len(workbook.sheetnames) == 1:
        return workbook.sheetnames[0]
    raise ValueError(
        f"Sheet {text!r} not found. Available sheets: {workbook.sheetnames}"
    )


def _sheet_bbox(sheet: Any, *, max_scan_rows: int = 20000) -> dict[str, Any]:
    min_row = min_col = None
    max_row = max_col = 0
    non_empty = 0
    rows_with_data = 0
    cols_with_data: set[int] = set()
    scanned_rows = 0
    scan_truncated = False
    for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        scanned_rows = row_idx
        if row_idx > max(1, max_scan_rows):
            scan_truncated = True
            break
        row_has_data = False
        for col_idx, value in enumerate(row, start=1):
            if value is None or str(value).strip() == "":
                continue
            non_empty += 1
            row_has_data = True
            cols_with_data.add(col_idx)
            min_row = row_idx if min_row is None else min(min_row, row_idx)
            min_col = col_idx if min_col is None else min(min_col, col_idx)
            max_row = max(max_row, row_idx)
            max_col = max(max_col, col_idx)
        if row_has_data:
            rows_with_data += 1
    if min_row is None:
        return {
            "min_row": None,
            "max_row": None,
            "min_col": None,
            "max_col": None,
            "range": None,
            "non_empty_cells": 0,
            "rows_with_data": 0,
            "cols_with_data": 0,
            "scanned_rows": scanned_rows,
            "scan_truncated": scan_truncated,
        }
    get_column_letter = _get_column_letter()
    return {
        "min_row": min_row,
        "max_row": max_row,
        "min_col": min_col,
        "max_col": max_col,
        "range": f"{get_column_letter(min_col)}{min_row}:"
        f"{get_column_letter(max_col)}{max_row}",
        "non_empty_cells": non_empty,
        "rows_with_data": rows_with_data,
        "cols_with_data": len(cols_with_data),
        "scanned_rows": scanned_rows,
        "scan_truncated": scan_truncated,
    }


def _read_row_values(
    sheet: Any, row_number: int, *, min_col: int = 1, max_col: int | None = None
) -> list[Any]:
    max_col = max_col or _safe_max_col(sheet)
    return [
        _clean_value(value)
        for row in sheet.iter_rows(
            min_row=row_number,
            max_row=row_number,
            min_col=min_col,
            max_col=max_col,
            values_only=True,
        )
        for value in row
    ]


def _unique_headers(values: Sequence[Any]) -> list[str]:
    get_column_letter = _get_column_letter()
    headers: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(values, start=1):
        name = (
            str(value).strip()
            if value is not None and str(value).strip()
            else get_column_letter(idx)
        )
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        headers.append(name)
    return headers


def _resolve_column(column: Any, headers: Sequence[str]) -> int:
    column_index_from_string = _column_index_from_string()
    if isinstance(column, int):
        return column
    text = str(column).strip()
    for idx, header in enumerate(headers, start=1):
        if str(header).strip() == text:
            return idx
    lowered = text.lower()
    for idx, header in enumerate(headers, start=1):
        if str(header).strip().lower() == lowered:
            return idx
    if re.fullmatch(r"[A-Za-z]{1,3}", text):
        return column_index_from_string(text.upper())
    raise ValueError(f"Column {column!r} not found in headers: {list(headers)}")


def _can_resolve_column(column: Any, headers: Sequence[str]) -> bool:
    try:
        _resolve_column(column, headers)
    except Exception:  # noqa: BLE001 - diagnostic helper
        return False
    return True


def _unresolved_columns(columns: Sequence[Any], headers: Sequence[str]) -> list[Any]:
    missing = []
    for column in columns:
        if column in (None, ""):
            continue
        if not _can_resolve_column(column, headers):
            missing.append(column)
    return missing


def _column_candidates(
    column: Any,
    headers: Sequence[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    get_column_letter = _get_column_letter()
    target = str(column).strip().lower()
    target_norm = _normalize_name(target)
    candidates = []
    for idx, header in enumerate(headers, start=1):
        header_text = str(header).strip()
        header_lower = header_text.lower()
        header_norm = _normalize_name(header_text)
        score = 0
        if header_lower == target:
            score = 100
        elif header_norm == target_norm and target_norm:
            score = 95
        elif target and target in header_lower:
            score = 80
        elif header_lower and header_lower in target:
            score = 70
        elif target_norm and target_norm in header_norm:
            score = 60
        if score:
            candidates.append(
                {
                    "header": header,
                    "column": get_column_letter(idx),
                    "index": idx,
                    "score": score,
                }
            )
    return sorted(candidates, key=lambda item: (-item["score"], item["index"]))[:limit]


def _collect_requested_columns(
    args: Mapping[str, Any],
    *,
    include_return_columns: bool = True,
) -> list[Any]:
    requested = []
    filters = args.get("filters") or {}
    if isinstance(filters, Mapping):
        requested.extend(filters.keys())
    if include_return_columns:
        requested.extend(_as_list(args.get("columns")))
    requested.extend(_as_list(args.get("group_by")))
    metrics = args.get("metrics") or []
    if isinstance(metrics, Mapping):
        metrics = [metrics]
    if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        for metric in metrics:
            if isinstance(metric, Mapping):
                metric_mapping = cast("Mapping[str, Any]", metric)
                if metric_mapping.get("column") not in (None, ""):
                    requested.append(metric_mapping.get("column"))
            elif not isinstance(metric, Mapping) and metric not in (None, ""):
                requested.append(metric)
    deduped = []
    seen = set()
    for item in requested:
        if item in (None, ""):
            continue
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _find_header_row_for_columns(
    sheet: Any,
    requested_columns: Sequence[Any],
    *,
    max_rows: int = 30,
) -> dict[str, Any] | None:
    columns = [column for column in requested_columns if column not in (None, "")]
    if not columns:
        return None
    best = None
    limit = min(_safe_max_row(sheet), max(1, int(max_rows or 30)))
    for row_idx in range(1, limit + 1):
        headers = _unique_headers(_read_row_values(sheet, row_idx))
        missing = _unresolved_columns(columns, headers)
        non_empty = sum(
            1 for header in headers if header is not None and str(header).strip()
        )
        score = len(columns) - len(missing)
        if score <= 0:
            continue
        item: dict[str, Any] = {
            "row": row_idx,
            "score": score,
            "requested_count": len(columns),
            "missing": missing,
            "headers": headers[:50],
            "non_empty_headers": non_empty,
        }
        if best is None or (item["score"], item["non_empty_headers"]) > (
            best["score"],
            best["non_empty_headers"],
        ):
            best = item
        if not missing:
            break
    return best


def _apply_header_auto_correction(
    sheet: Any,
    header_row: int,
    requested_columns: Sequence[Any],
) -> tuple[int, list[str], dict[str, Any] | None]:
    headers = _unique_headers(_read_row_values(sheet, header_row))
    missing = _unresolved_columns(requested_columns, headers)
    if not missing:
        return header_row, headers, None
    candidate = _find_header_row_for_columns(sheet, requested_columns)
    if candidate and candidate["row"] != header_row:
        corrected_headers = _unique_headers(_read_row_values(sheet, candidate["row"]))
        corrected_missing = _unresolved_columns(requested_columns, corrected_headers)
        if len(corrected_missing) < len(missing):
            return (
                int(candidate["row"]),
                corrected_headers,
                {
                    "from_header_row": header_row,
                    "to_header_row": candidate["row"],
                    "original_missing_columns": missing,
                    "remaining_missing_columns": corrected_missing,
                    "candidate": candidate,
                },
            )
    return header_row, headers, None


def _header_row_candidates(
    sheet: Any,
    *,
    requested_columns: Sequence[Any] | None = None,
    max_rows: int = 30,
    limit: int = 8,
) -> list[dict[str, Any]]:
    columns = list(requested_columns or [])
    candidates = []
    max_row = min(_safe_max_row(sheet), max(1, int(max_rows or 30)))
    for row_idx in range(1, max_row + 1):
        headers = _unique_headers(_read_row_values(sheet, row_idx))
        non_empty = sum(
            1 for header in headers if header is not None and str(header).strip()
        )
        if not non_empty:
            continue
        missing = _unresolved_columns(columns, headers) if columns else []
        score = (len(columns) - len(missing)) if columns else min(non_empty, 10)
        item: dict[str, Any] = {
            "row": row_idx,
            "score": score,
            "non_empty_headers": non_empty,
            "missing_requested_columns": missing,
            "headers": headers[:50],
        }
        if missing:
            item["column_candidates"] = {
                str(column): _column_candidates(column, headers) for column in missing
            }
        candidates.append(item)
    return sorted(candidates, key=lambda item: (-item["score"], item["row"]))[:limit]


def _row_matches(
    row_values: Sequence[Any],
    filters: Mapping[str, Any],
    headers: Sequence[str],
    case_sensitive: bool,
) -> bool:
    for column, condition in filters.items():
        idx = _resolve_column(column, headers)
        value = row_values[idx - 1] if idx - 1 < len(row_values) else None
        if not _value_matches(value, condition, case_sensitive):
            return False
    return True


def _value_matches(value: Any, condition: Any, case_sensitive: bool) -> bool:
    if isinstance(condition, Mapping):
        if condition.get("non_empty") is True and (
            value is None or str(value).strip() == ""
        ):
            return False
        if (
            condition.get("non_empty") is False
            and value is not None
            and str(value).strip() != ""
        ):
            return False
        value_text = _text_norm(value, case_sensitive)
        if "equals" in condition and value_text != _text_norm(
            condition.get("equals"), case_sensitive
        ):
            return False
        if (
            "contains" in condition
            and _text_norm(condition.get("contains"), case_sensitive) not in value_text
        ):
            return False
        number = _to_number(value)
        for op, fn in (
            ("gt", lambda a, b: a > b),
            ("ge", lambda a, b: a >= b),
            ("lt", lambda a, b: a < b),
            ("le", lambda a, b: a <= b),
        ):
            if op in condition:
                rhs = _to_number(condition.get(op))
                if number is None or rhs is None or not fn(number, rhs):
                    return False
        return True
    return _text_norm(value, case_sensitive) == _text_norm(condition, case_sensitive)


def _metric_spec(metric: Any, headers: Sequence[str]) -> dict[str, Any]:
    if not isinstance(metric, Mapping):
        metric = {"column": metric, "op": "sum"}
    op = str(metric.get("op", "sum")).lower()
    column = metric.get("column")
    idx = (
        _resolve_column(column, headers)
        if column is not None and op != "count"
        else None
    )
    name = metric.get("name") or (f"{op}_{column}" if column is not None else op)
    return {"op": op, "column": column, "idx": idx, "name": str(name)}


def _update_metric(
    state: dict[str, Any], spec: Mapping[str, Any], row: Sequence[Any]
) -> None:
    op = str(spec["op"])
    if op == "count":
        state["count"] += 1
        return
    idx = spec.get("idx")
    value = row[idx - 1] if isinstance(idx, int) and idx - 1 < len(row) else None
    if op == "count_non_empty":
        if value is not None and str(value).strip():
            state["count"] += 1
        return
    number = _to_number(value)
    if number is None:
        return
    state["count"] += 1
    state["sum"] += number
    state["min"] = number if state["min"] is None else min(state["min"], number)
    state["max"] = number if state["max"] is None else max(state["max"], number)


def _finish_metric(state: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    op = str(spec["op"])
    if op in {"count", "count_non_empty"}:
        return state["count"]
    if op == "sum":
        return state["sum"]
    if op in {"avg", "mean"}:
        return state["sum"] / state["count"] if state["count"] else None
    if op == "min":
        return state["min"]
    if op == "max":
        return state["max"]
    return dict(state)


def _excel_error_diagnostics(
    args: Mapping[str, Any], *, workspace: Path, references: Path
) -> dict[str, Any]:
    try:
        path = _resolve_workbook_path(args, workspace=workspace, references=references)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"path_error": f"{type(exc).__name__}: {exc}"}
    try:
        load_workbook = _load_workbook()
        workbook = load_workbook(filename=path, read_only=True, data_only=True)
        requested_sheet = args.get("sheet_name") or args.get("sheet")
        requested_columns = _collect_requested_columns(args)
        diagnostics: dict[str, Any] = {
            "resolved_path": str(path),
            "available_sheets": workbook.sheetnames,
            "requested_sheet": requested_sheet,
        }
        if requested_columns:
            diagnostics["requested_columns"] = requested_columns
        sheet_names = []
        try:
            sheet_names.append(_choose_sheet_name(workbook, requested_sheet))
        except Exception:  # noqa: BLE001 - diagnostics should not mask primary error
            sheet_names.extend(workbook.sheetnames[:5])
        samples = []
        for sheet_name in sheet_names[:5]:
            sheet = workbook[sheet_name]
            samples.append(
                {
                    "sheet_name": sheet_name,
                    "header_row_candidates": _header_row_candidates(
                        sheet,
                        requested_columns=requested_columns,
                    ),
                }
            )
        diagnostics["sheet_diagnostics"] = samples
        return diagnostics
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"resolved_path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _clean_value(value: Any, *, max_len: int = 500) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, int | float | bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    text = str(value)
    return text[:max_len] + "...[truncated]" if len(text) > max_len else text


def _text_norm(value: Any, case_sensitive: bool) -> str:
    text = "" if value is None else str(value).strip()
    return text if case_sensitive else text.lower()


def _to_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _safe_max_row(sheet: Any) -> int:
    return int(sheet.max_row or 1)


def _safe_max_col(sheet: Any) -> int:
    return int(sheet.max_column or 1)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_name_tokens(value: str) -> str:
    return " ".join(sorted(re.findall(r"[a-z0-9]+", value.lower())))


def _workbook_path_priority(
    path: Path, *, label_prefix: str = ""
) -> tuple[int, int, str]:
    text = str(path)
    if label_prefix == "A":
        buckets = ("deliverable_task_id_files", "reference_task_id_files", "workdir")
    elif label_prefix == "B":
        buckets = ("workdir", "deliverable_task_id_files", "reference_task_id_files")
    else:
        buckets = ("workdir", "deliverable_task_id_files", "reference_task_id_files")
    bucket_rank = next(
        (idx for idx, marker in enumerate(buckets) if marker in text),
        len(buckets),
    )
    return (bucket_rank, len(text), text)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
