"""Model-driven whole-program meta-strategy (the Plan-2 LLM-strategy seam).

Benchmark-agnostic: a model rewrites full files under a path prefix (AST-validated
for ``.py``), returning a Proposal. The prompt and prefix are injected, so this
component carries no benchmark specifics. Recipes supply the domain prompt.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from simple_agent_lab.evolution import Proposal
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import Context, Run, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

DEFAULT_SYSTEM_PROMPT = """You are a meta-agent evolving a program.

The program is a set of files under a fixed path prefix. You may edit any file
under that prefix, add new files under it, or remove one (set its value to null).
Provide FULL new file content, never a diff.

Return ONLY JSON: {"note": "...", "evidence": ["..."],
"edits": {"<prefix>/<path>": "FULL content" | null}}

Make one focused change likely to improve the measured objective.
"""

JSON_REPAIR_MAX_ATTEMPTS = 3


def model_program_strategy(
    *,
    provider: Provider,
    surface: AgentSurface | None = None,
    editable_components: Sequence[str] = ("everything",),
    prefix: str = "agent/",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    parent_selection: str = "current",
    parent_selector: Callable[[Context, str], str] | None = None,
    build_prompt: Callable[[Version, Sequence[Run]], str] | None = None,
    complete_fn: Callable[[LLMRequest], Any] = complete,
    max_tokens: int = 4000,
    kind: str = "code",
    json_repair_attempts: int = JSON_REPAIR_MAX_ATTEMPTS,
    log_fn: Callable[[str], None] | None = None,
) -> Callable[[Context], Proposal | None]:
    """Return a ``(Context) -> Proposal`` strategy that rewrites files under prefix."""

    prompt_builder = build_prompt or (lambda v, f: _default_prompt(v, f, prefix))
    surface_brief = ""
    if surface is not None:
        surface_brief = "\n\n" + surface.prompt_brief(components=editable_components)
    effective_system_prompt = system_prompt + surface_brief

    def strategy(ctx: Context) -> Proposal | None:
        parent = _select_parent(
            ctx, parent_selection=parent_selection, parent_selector=parent_selector
        )
        base = ctx.version(parent)
        messages = [llm_message("user", prompt_builder(base, ctx.failures))]
        payload = None
        attempts = max(1, int(json_repair_attempts))
        for attempt in range(1, attempts + 1):
            response = complete_fn(
                LLMRequest(
                    provider=provider,
                    messages=list(messages),
                    system_prompt=effective_system_prompt,
                    max_tokens=max_tokens,
                )
            )
            try:
                payload = parse_model_json(response.text)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                _log(
                    log_fn,
                    "meta-strategy returned invalid JSON "
                    f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {exc}",
                )
                if attempt >= attempts:
                    return None
                messages.append(_json_repair_message(response.text, exc))
        if payload is None:
            return None
        raw_edits = payload.get("edits", {})
        if surface is not None:
            if isinstance(raw_edits, Mapping):
                validated = surface.validate_edits(
                    raw_edits,
                    components=editable_components,
                )
                edits = validated.edits
                rejected = validated.rejected
            else:
                edits, rejected = {}, ()
        else:
            edits, rejected = safe_prefix_edits(raw_edits, prefix=prefix)
        evidence = tuple(str(x) for x in payload.get("evidence", ()))
        evidence += tuple(f"discarded-disallowed-path:{p}" for p in rejected)
        if not edits:
            detail = f"; rejected={', '.join(rejected)}" if rejected else ""
            _log(log_fn, f"meta-strategy produced no valid edits{detail}")
            return None
        return Proposal(
            base=parent,
            edits=edits,
            note=str(payload.get("note", "model program edit")),
            evidence=evidence,
            kind=kind,
        )

    return strategy


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse plain or fenced JSON from the model."""

    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("model response must be a JSON object")
    return data


def _json_repair_message(text: str, exc: BaseException):
    preview = text.strip().replace("\n", "\\n")[:500] or "<empty>"
    return llm_message(
        "system",
        "Your previous meta-strategy reply was not valid JSON "
        f"({type(exc).__name__}: {exc}). The reply began: {preview!r}. "
        "Return only a JSON object with note, evidence, and edits; do not include "
        "Markdown or explanatory text.",
    )


def safe_prefix_edits(
    raw_edits: Any, *, prefix: str
) -> tuple[dict[str, str | None], tuple[str, ...]]:
    """Keep edits under ``prefix`` (str content or None tombstone); reject the rest."""

    if not isinstance(raw_edits, Mapping):
        return {}, ()
    edits: dict[str, str | None] = {}
    rejected: list[str] = []
    for raw_path, content in raw_edits.items():
        path = str(raw_path)
        if not _prefix_path_ok(path, prefix):
            rejected.append(path)
        elif content is None:
            edits[path] = None
        elif isinstance(content, str) and _python_ok(path, content):
            edits[path] = content
        else:
            rejected.append(path)
    return edits, tuple(rejected)


def _prefix_path_ok(path: str, prefix: str) -> bool:
    if path.startswith("/") or not path.startswith(prefix):
        return False
    return ".." not in PurePosixPath(path).parts


def _python_ok(path: str, content: str) -> bool:
    if not path.endswith(".py"):
        return True
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


def _select_parent(
    ctx: Context,
    *,
    parent_selection: str,
    parent_selector: Callable[[Context, str], str] | None = None,
) -> str:
    if parent_selection == "current":
        return ctx.current.hash
    if parent_selector is None:
        raise ValueError(
            "non-current parent selection requires a recipe-provided parent_selector"
        )
    return parent_selector(ctx, parent_selection) or ctx.current.hash


def _default_prompt(version: Version, failures: Sequence[Run], prefix: str) -> str:
    files = [n for n in version.files() if n.startswith(prefix)]
    program = "\n\n".join(f"### {n}\n{version.read(n)}" for n in files) or "- (empty)"
    fail = (
        "\n".join(f"- {r.instance_id}: reward={r.reward}" for r in failures) or "- none"
    )
    return (
        f"Current program (version {version.hash}, prefix {prefix!r}):\n{program}\n\n"
        f"Failing runs:\n{fail}\n\n"
        "Propose one focused edit. Return full file contents as JSON."
    )


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)
        return
    print(message, file=sys.stderr, flush=True)
