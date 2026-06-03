"""Load GDPVal rows into Simple Agent Lab instance dictionaries."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASET = "openai/gdpval"
DEFAULT_HF_SPLIT = "train"


def load_instances(
    path: str | Path | None = None,
    *,
    task_ids: Iterable[str] | None = None,
    limit: int | None = None,
    require_deliverables: bool = True,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_split: str = DEFAULT_HF_SPLIT,
    hf_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load GDPVal rows from Hugging Face or a local JSONL/JSON/Parquet file.

    Hugging Face support is lazy: install ``datasets`` in the caller
    environment when reading ``openai/gdpval`` directly. Parquet support is also
    lazy and requires ``pyarrow``. The framework itself does not add either as a
    hard dependency.
    """

    wanted = {str(task_id) for task_id in task_ids or []}
    rows: list[dict[str, Any]] = []
    raw_rows = (
        _read_huggingface_rows(
            dataset=hf_dataset,
            split=hf_split,
            cache_dir=hf_cache_dir,
        )
        if path is None
        else _read_rows(Path(path))
    )
    for row in raw_rows:
        normalized = _normalize_row(row)
        if wanted and str(normalized["instance_id"]) not in wanted:
            continue
        if require_deliverables and not _has_deliverable_files(normalized):
            continue
        rows.append(normalized)
        if limit is not None and len(rows) >= max(0, limit):
            break
    return rows


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield json.loads(text)
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, Mapping))
            return
        if isinstance(value, Mapping):
            data = value.get("data") or value.get("instances") or value.get("rows")
            if isinstance(data, list):
                yield from (row for row in data if isinstance(row, Mapping))
                return
            yield value
            return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "Reading GDPVal parquet input requires pyarrow in the caller "
                "environment. Install it there or convert the file to JSONL."
            ) from exc
        yield from pq.read_table(path).to_pylist()
        return
    raise ValueError(f"Unsupported GDPVal input file type: {path}")


def _read_huggingface_rows(
    *,
    dataset: str,
    split: str,
    cache_dir: str | Path | None,
) -> Iterable[Mapping[str, Any]]:
    resolved_cache_dir = None
    if cache_dir is not None:
        resolved_cache_dir = str(Path(cache_dir).resolve())
        os.environ["HF_HOME"] = resolved_cache_dir
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Reading GDPVal from Hugging Face requires the 'datasets' package. "
            "Run with: uv run --with datasets python runs/run_gdpval.py ..."
        ) from exc
    dataset_rows = load_dataset(
        dataset,
        split=split,
        streaming=True,
        cache_dir=resolved_cache_dir,
    )
    yield from dataset_rows


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(
        row.get("instance_id")
        or row.get("task_id")
        or row.get("id")
        or row.get("taskId")
        or ""
    ).strip()
    if not task_id:
        raise ValueError(f"GDPVal row is missing task id: {row!r}")
    prompt = str(row.get("prompt_en") or row.get("prompt") or row.get("question") or "")
    return {
        **dict(row),
        "instance_id": task_id,
        "task_id": task_id,
        "prompt": prompt,
    }


def _has_deliverable_files(row: Mapping[str, Any]) -> bool:
    return bool(_as_list_like(row.get("deliverable_files")))


def _as_list_like(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [value]
            return _as_list_like(parsed)
        return [value]
    if isinstance(value, Mapping):
        return [item for item in value.values() if item]
    if isinstance(value, (list, tuple)):
        return [item for item in value if item]
    return [value]
