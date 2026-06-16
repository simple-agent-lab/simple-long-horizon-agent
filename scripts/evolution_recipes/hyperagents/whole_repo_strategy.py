"""Whole-repo coding strategy: the meta-agent edits the agent program (Path B)."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from simple_agent_lab.evolution import Proposal, archive
from simple_agent_lab.evolution.types import Context, Run, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

from .sal_meta_strategy import parse_model_json

PACKAGE_PREFIX = "agent/"

SYSTEM_PROMPT = """You are a meta-agent evolving a SWE-bench coding agent.

The agent is a Python package under `agent/`. Its entry module
`agent/agent_program.py` defines:

    def build_agent(*, provider, cwd, base_system_prompt) -> Agent

You may edit any file under `agent/`, add new files under `agent/`, or remove a
file (set its value to null). Provide FULL new file content, never a diff. The
package may import the installed `simple_agent_lab` wheel (tools, llm_agent,
agents.starter). Keep `build_agent` present and returning an Agent.

Return ONLY JSON: {"note": "...", "evidence": ["..."],
"edits": {"agent/<path>": "FULL content" | null}}

Make one focused change likely to raise the SWE-bench resolve rate (e.g. a
reproduce-then-fix loop, a verification step before finishing, an extra tool).
"""


def safe_package_edits(
    raw_edits: Any,
) -> tuple[dict[str, str | None], tuple[str, ...]]:
    """Keep edits under ``agent/`` (str content or None tombstone); reject rest."""

    if not isinstance(raw_edits, Mapping):
        return {}, ()
    edits: dict[str, str | None] = {}
    rejected: list[str] = []
    for raw_path, content in raw_edits.items():
        path = str(raw_path)
        if not _package_path_ok(path):
            rejected.append(path)
            continue
        if content is None:
            edits[path] = None
            continue
        if isinstance(content, str) and _python_ok(path, content):
            edits[path] = content
        else:
            rejected.append(path)
    return edits, tuple(rejected)


def _package_path_ok(path: str) -> bool:
    if path.startswith("/") or not path.startswith(PACKAGE_PREFIX):
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


def make_strategy(
    workspace: str | Path,
    *,
    provider: Provider,
    parent_selection: str = "score_child_prop",
    complete_fn: Callable[[LLMRequest], Any] = complete,
):
    """Return a `(Context) -> Proposal` strategy that rewrites the agent package."""

    workspace = Path(workspace)

    def strategy(ctx: Context) -> Proposal:
        parent = _select_parent(workspace, ctx, parent_selection=parent_selection)
        base = ctx.version(parent) if parent else ctx.current
        prompt = _build_prompt(base, ctx.failures)
        response = complete_fn(
            LLMRequest(
                provider=provider,
                messages=[llm_message("user", prompt)],
                system_prompt=SYSTEM_PROMPT,
                max_tokens=4000,
            )
        )
        payload = parse_model_json(response.text)
        edits, rejected = safe_package_edits(payload.get("edits", {}))
        evidence = tuple(str(x) for x in payload.get("evidence", ()))
        evidence += tuple(f"discarded-disallowed-path:{p}" for p in rejected)
        return Proposal(
            base=parent,
            edits=edits,
            note=str(payload.get("note", "whole-repo agent edit")),
            evidence=evidence,
            kind="code",
        )

    return strategy


def _select_parent(workspace: Path, ctx: Context, *, parent_selection: str) -> str:
    nodes = archive.nodes(workspace)
    if not nodes:
        return ctx.current.hash
    try:
        return archive.select_parent(nodes, method=parent_selection)
    except ValueError:
        return ctx.current.hash


def _build_prompt(version: Version, failures: Sequence[Run]) -> str:
    files = [n for n in version.files() if n.startswith(PACKAGE_PREFIX)]
    package = "\n\n".join(
        f"### {name}\n{version.read(name)}" for name in files
    ) or "- (no package files; default applies)"
    fail = "\n".join(
        f"- {r.instance_id}: reward={r.reward}" for r in failures
    ) or "- no failures"
    return (
        f"Current agent package (version {version.hash}):\n{package}\n\n"
        f"Failed train runs:\n{fail}\n\n"
        "Propose one focused edit to the agent package. Return full file contents."
    )
