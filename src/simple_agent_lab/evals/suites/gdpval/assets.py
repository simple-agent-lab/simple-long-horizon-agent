"""Reference-file staging helpers for GDPVal."""

from __future__ import annotations

import base64
import hashlib
import ast
import json
import os
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def normalize_reference_file_inputs(
    raw: Any,
    *,
    reference_root: str | Path | None = None,
    task_id: str = "",
) -> list[dict[str, Any]]:
    """Normalize GDPVal reference-file fields into container-staged blobs.

    The public rows may carry JSON strings, Python-literal strings, lists, or
    dictionaries. This helper accepts local file paths from any of those shapes
    and returns JSON-safe descriptors with base64 payloads. The model never sees
    these descriptors directly; the container half writes them under
    ``REFERENCE_DIR`` and only exposes the sandbox paths in the task prompt.
    """

    values = _flatten_reference_values(_coerce_json_like(raw))
    root = Path(reference_root).resolve() if reference_root else None
    descriptors: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            embedded = _embedded_payload_descriptor(value)
            if embedded is not None:
                descriptors.append(embedded)
                continue
            source, label = _mapping_source_and_label(value)
        else:
            source = str(value).strip()
            label = source
        if not source:
            continue
        if _is_remote_source(source):
            downloaded = _download_reference_descriptor(source, label)
            descriptors.append(downloaded)
            continue
        path = _resolve_reference_path(source, root, task_id=task_id)
        if path is None or not path.is_file():
            descriptors.append(
                {
                    "name": _safe_reference_name(label or source),
                    "source": source,
                    "missing": True,
                }
            )
            continue
        data = path.read_bytes()
        descriptors.append(
            {
                "name": _safe_reference_name(label or path.name),
                "source": source,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    return descriptors


def write_reference_files(
    descriptors: Iterable[Mapping[str, Any]],
    reference_dir: str | Path,
) -> list[dict[str, Any]]:
    """Write normalized reference descriptors to ``reference_dir``."""

    root = Path(reference_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for descriptor in descriptors:
        name = _safe_reference_name(str(descriptor.get("name") or "reference"))
        entry: dict[str, Any] = {
            "name": name,
            "source": str(descriptor.get("source") or ""),
            "missing": bool(descriptor.get("missing")),
        }
        payload = descriptor.get("data_base64")
        if isinstance(payload, str) and payload:
            data = base64.b64decode(payload)
            path = _safe_join(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entry.update(
                {
                    "path": str(path),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        elif "text" in descriptor or "content" in descriptor:
            text = str(descriptor.get("text", descriptor.get("content", "")))
            data = text.encode("utf-8")
            path = _safe_join(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entry.update(
                {
                    "path": str(path),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        manifest.append(entry)
    (root / "_sal_reference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
                pass
    return value


def _flatten_reference_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if _looks_like_file_descriptor(value):
            return [value]
        flattened: list[Any] = []
        for key, item in value.items():
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            elif isinstance(item, Mapping):
                flattened.append({"name": str(key), **dict(item)})
            else:
                flattened.append({"name": str(key), "path": item})
        return flattened
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _looks_like_file_descriptor(value: Mapping[str, Any]) -> bool:
    keys = set(value)
    return bool(
        keys
        & {
            "path",
            "file",
            "url",
            "source",
            "data_base64",
            "text",
            "content",
        }
    )


def _embedded_payload_descriptor(value: Mapping[str, Any]) -> dict[str, Any] | None:
    name = _safe_reference_name(
        str(value.get("name") or value.get("path") or "reference")
    )
    source = str(value.get("source") or value.get("path") or value.get("url") or "")
    if isinstance(value.get("data_base64"), str) and value.get("data_base64"):
        data = base64.b64decode(str(value["data_base64"]))
        return {
            "name": name,
            "source": source,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    if "text" in value or "content" in value:
        text = str(value.get("text", value.get("content", "")))
        data = text.encode("utf-8")
        return {
            "name": name,
            "source": source,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    return None


def _resolve_reference_path(
    source: str, root: Path | None, *, task_id: str = ""
) -> Path | None:
    if source.startswith(("http://", "https://", "tos://")):
        return None
    path = Path(source)
    if path.is_absolute():
        return path
    if root is not None:
        candidate = root / path
        if candidate.is_file():
            return candidate
        basename = Path(source.replace("\\", "/")).name
        if task_id and basename:
            fallback = root / task_id / basename
            if fallback.is_file():
                return fallback
        return candidate
    return path


def _mapping_source_and_label(value: Mapping[str, Any]) -> tuple[str, str]:
    url = str(value.get("url") or value.get("source_url") or "").strip()
    path = str(
        value.get("path")
        or value.get("file")
        or value.get("source")
        or value.get("name")
        or ""
    ).strip()
    source = url if _is_remote_source(url) else path or url
    label = str(value.get("name") or value.get("label") or path or source).strip()
    return source, label


def _is_remote_source(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _download_reference_descriptor(source: str, label: str) -> dict[str, Any]:
    name = _safe_reference_name(label or _url_basename(source))
    try:
        with urllib.request.urlopen(source, timeout=120) as response:
            data = response.read()
    except Exception:
        return {
            "name": name,
            "source": source,
            "missing": True,
        }
    return {
        "name": name,
        "source": source,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


def _url_basename(source: str) -> str:
    path = urlparse(source).path
    return unquote(Path(path).name) or "reference"


def _safe_reference_name(name: str) -> str:
    raw = name.strip().replace("\\", "/").strip("/")
    if not raw:
        return "reference"
    parts = [
        "".join(c if c.isalnum() or c in "._-" else "_" for c in part)
        for part in raw.split("/")
        if part and part not in {".", ".."}
    ]
    safe = "/".join(parts) or "reference"
    if safe != raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        stem, suffix = os.path.splitext(safe)
        safe = f"{stem or 'reference'}-{digest}{suffix}"
    return safe


def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    base = root.resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"reference path escapes root: {rel!r}")
    return target
