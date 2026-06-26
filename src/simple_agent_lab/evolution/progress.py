"""Small stdout progress reporter for self-evolving experiments."""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO


class ProgressReporter:
    """Format stable one-line progress records."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enabled: bool = True,
        max_value_chars: int = 160,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = enabled
        self.max_value_chars = max(8, int(max_value_chars))
        self._lock = threading.Lock()

    def line(self, event: str, *parts: object, **fields: object) -> None:
        with self._lock:
            if not self.enabled:
                return
            try:
                tokens = ["[progress]", str(event)]
                tokens.extend(
                    str(part) for part in parts if part is not None and part != ""
                )
                for key, value in fields.items():
                    if value is None:
                        continue
                    tokens.append(f"{key}={self._format_value(value, key=key)}")
                print(" ".join(tokens), file=self.stream, flush=True)
            except Exception:
                self.enabled = False

    def _format_value(self, value: object, *, key: str = "") -> str:
        if isinstance(value, float):
            return _format_float(value, signed=key == "delta")
        if isinstance(value, Path):
            text = str(value)
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        text = _single_line(text)
        if len(text) > self.max_value_chars:
            text = text[: self.max_value_chars - 3] + "..."
        if not text or any(ch.isspace() for ch in text) or '"' in text:
            text = text.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{text}"'
        return text


def mean_score(
    scores: Mapping[str, Mapping[str, float]], *, dim: str = "reward"
) -> float | None:
    values = [float(per_dim[dim]) for per_dim in scores.values() if dim in per_dim]
    if not values:
        return None
    return sum(values) / len(values)


def signed_delta(base: float | None, candidate: float | None) -> float | None:
    if base is None or candidate is None:
        return None
    return candidate - base


def _format_float(value: float, *, signed: bool = False) -> str:
    if signed and value > 0:
        sign = "+"
        return f"{sign}{value:.3f}"
    return f"{value:.3f}"


def _single_line(value: str) -> str:
    return " ".join(value.split())
