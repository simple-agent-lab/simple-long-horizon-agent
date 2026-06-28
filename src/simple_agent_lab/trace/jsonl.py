"""JSON-safe coercion and (atomic) JSONL read/write helpers.

This is the lowest layer of the trajectory package: pure serialization
plumbing with no knowledge of spans, runs, or the live session. Everything
that persists a trajectory ultimately goes through ``json_safe`` here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO


def json_safe(value: Any) -> Any:
    """Convert project dataclasses and containers into JSON-safe values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSON records from a file.

    Supports both single-line JSONL and pretty-printed (``indent=2``) records.
    """
    text = Path(path).read_text(encoding="utf-8")
    return _parse_json_records(text)


def _parse_json_records(text: str) -> list[dict[str, Any]]:
    """Extract all top-level JSON objects from *text* using incremental decoding."""
    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        idx = _skip_whitespace(text, idx, length)
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        records.append(obj)
        idx = end
    return records


def _skip_whitespace(text: str, idx: int, length: int) -> int:
    while idx < length and text[idx] in " \t\n\r":
        idx += 1
    return idx


def _dump_records(f: TextIO, records: Iterable[Mapping[str, Any]]) -> None:
    """Write each record as one compact JSON line (true JSONL).

    The single source of truth for the on-disk shape shared by
    :func:`write_jsonl` and :func:`write_jsonl_atomic`. One record per line is
    what the v5 trajectory stream (header line + one line per event) and the raw
    pool (one blob per line) need; readers (`read_jsonl`) still accept old
    pretty-printed records too.
    """
    for record in records:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        _dump_records(f, records)


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Write ``records`` to ``path`` via tmp-file + rename so readers never see a torn file.

    The same on-disk shape as :func:`write_jsonl` (pretty-printed JSON,
    one record per top-level object); only the write strategy differs.
    Used by the incremental writer so a polling viewer reading the path
    mid-run always observes a complete record.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.part")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            _dump_records(f, records)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync is best-effort; e.g. on tmpfs it may not be supported.
                pass
        os.replace(tmp, out)
    except BaseException:
        # Make sure we never leave a stray ``.part`` file behind on error.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
