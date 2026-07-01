"""Best-effort JSON-object extraction from LLM text output.

Models sometimes wrap JSON in prose or a fenced code block instead of
returning it bare; this is the one place that tolerance lives so every
caller parsing a structured LLM response shares the same rules.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first top-level JSON object out of *text*, or return ``None``.

    Strips a leading/trailing ```` ``` ```` or ```` ```json ```` fence, then
    falls back to the outermost ``{...}`` substring if the whole text does not
    parse as-is (handles prose like ``"Here you go: {...}"``). Returns
    ``None`` — never raises — when no JSON object can be recovered, including
    when the parsed value is valid JSON but not an object (e.g. ``true``).
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
