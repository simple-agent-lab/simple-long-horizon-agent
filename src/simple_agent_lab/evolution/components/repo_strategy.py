"""Agentic source-tree meta-strategy helpers."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.evolution.source_tree import (
    SOURCE_ROOT,
    cheap_validate_source_tree,
    source_tree_agent_surface,
    validate_source_tree_edits,
)
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import Context, Proposal, Version

DEFAULT_SOURCE_TREE_AGENT_PROMPT = """You are a meta-agent improving Simple Agent Lab.

You may inspect and edit this temporary repository copy with bash. Make one
small, focused change under src/simple_agent_lab/ that is likely to improve the
current self-evolving agent system. Do not edit recipes or config migration in
this task.
"""

DEFAULT_SOURCE_TREE_AGENT_TASK = """Improve the source tree in this temporary copy.

Read SELF_EVOLUTION_CONTEXT.md first.
Use self_evolution/ for deeper local evidence when the briefing is insufficient.
Only changes under src/simple_agent_lab/ can become a proposal. Prefer small,
readable Python edits. When finished, reply with a short summary.
"""

_COPY_IGNORE_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

_META_EVIDENCE_DIR = "self_evolution"
_MAX_FAILURE_ARTIFACTS = 8
_MAX_TRACE_EVENTS = 8
_MAX_EVENT_TEXT = 500


class _RunnableAgent(Protocol):
    def run(self, task: str, *, max_turns: int) -> tuple[object, Iterator[object]]: ...


AgentBuilder = Callable[..., _RunnableAgent]
ValidationFn = Callable[[Path, Mapping[str, str]], None]


def proposal_from_candidate_tree(
    base_tree: Path,
    changed_tree: Path,
    *,
    base_hash: str,
    note: str,
    evidence: Sequence[str] = (),
) -> Proposal:
    """Turn changed ``src/simple_agent_lab/**/*.py`` files into a Proposal.

    Deletions are recorded as evidence and ignored for now. The current
    ``candidate_source_artifacts`` staging helper overlays files but cannot
    faithfully tombstone inherited source files.
    """

    edits: dict[str, str] = {}
    proposal_evidence = list(evidence)

    for rel in _changed_paths(base_tree, changed_tree):
        rel_text = rel.as_posix()
        base_path = base_tree / rel
        changed_path = changed_tree / rel

        if not _is_under_source_root(rel):
            proposal_evidence.append(f"discarded-outside-source:{rel_text}")
            continue
        if changed_path.is_symlink() or base_path.is_symlink():
            proposal_evidence.append(f"discarded-symlink-source:{rel_text}")
            continue
        if not changed_path.exists():
            proposal_evidence.append(f"discarded-deleted-source:{rel_text}")
            continue
        if changed_path.suffix != ".py":
            proposal_evidence.append(f"discarded-non-python-source:{rel_text}")
            continue
        if not changed_path.is_file():
            continue

        try:
            content = changed_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            proposal_evidence.append(f"discarded-non-utf8-source:{rel_text}")
            continue
        errors = validate_source_tree_edits({rel_text: content})
        if errors:
            joined = "; ".join(errors)
            proposal_evidence.append(f"discarded-invalid-source:{rel_text}: {joined}")
            continue
        edits[rel_text] = content

    return Proposal(
        edits=edits,
        note=note,
        evidence=tuple(proposal_evidence),
        base=base_hash,
        kind="source",
    )


def source_tree_agent_strategy(
    *,
    provider: Any,
    repo_root: Path,
    max_turns: int = 20,
    validation: ValidationFn = cheap_validate_source_tree,
    parent_selection: str = "current",
    parent_selector: Callable[[Context, str], str] | None = None,
    agent_builder: AgentBuilder | None = None,
    system_prompt: str = DEFAULT_SOURCE_TREE_AGENT_PROMPT,
    task: str = DEFAULT_SOURCE_TREE_AGENT_TASK,
    name: str = "source_tree_meta_agent",
    surface: AgentSurface | None = None,
    editable_components: Sequence[str] = (),
    proposal_kind: str = "source",
    evidence_marker: str = "source-tree-agent-ran",
) -> Callable[[Context], Proposal | None]:
    """Return a strategy that lets a bash-capable meta-agent edit a repo copy."""

    root = Path(repo_root)
    build_agent = agent_builder or make_bash_agent
    agent_surface = surface or source_tree_agent_surface(root)
    components = tuple(editable_components) or ("everything",)

    def strategy(ctx: Context) -> Proposal | None:
        parent = _select_parent(
            ctx, parent_selection=parent_selection, parent_selector=parent_selector
        )
        base_version = ctx.version(parent)
        with tempfile.TemporaryDirectory(prefix="sal-source-tree-") as tmp:
            base_tree = Path(tmp) / "base"
            candidate = Path(tmp) / "candidate"
            shutil.copytree(
                root,
                base_tree,
                symlinks=True,
                ignore=_copy_ignore,
            )
            _overlay_source_version(base_tree, base_version.files(), base_version.read)
            shutil.copytree(base_tree, candidate, symlinks=True)
            _write_meta_agent_context(base_tree, ctx, base_version.hash)
            _write_meta_agent_context(candidate, ctx, base_version.hash)

            agent = build_agent(
                provider=provider,
                cwd=candidate,
                name=name,
                system_prompt=system_prompt,
            )
            _state, events = agent.run(task, max_turns=max_turns)
            for _event in events:
                pass

            raw_edits = _changed_text_edits(base_tree, candidate)
            validated = agent_surface.validate_edits(raw_edits, components=components)
            source_edits = {
                path: content
                for path, content in validated.edits.items()
                if content is not None
            }
            deleted = tuple(
                path for path, content in validated.edits.items() if content is None
            )
            edits, unchanged = _content_changing_edits(base_version, source_edits)
            evidence = [evidence_marker]
            evidence.extend(
                f"discarded-disallowed-path:{path}" for path in validated.rejected
            )
            evidence.extend(f"discarded-deleted-source:{path}" for path in deleted)
            evidence.extend(f"discarded-unchanged-path:{path}" for path in unchanged)
            proposal = Proposal(
                edits=edits,
                note="source-tree meta-agent edit",
                evidence=tuple(evidence),
                base=parent,
                kind=proposal_kind,
            )
            if not proposal.edits:
                return None

            files = {
                path: content
                for path, content in proposal.edits.items()
                if isinstance(content, str)
                and not validate_source_tree_edits({path: content})
            }
            if not files:
                return None
            try:
                validation(base_tree, files)
            except Exception:
                return None
            return proposal

    return strategy


def _context_briefing(ctx: Context, parent_hash: str) -> str:
    lines = [
        "# Self Evolution Context",
        "",
        f"- Parent version: {parent_hash}",
        f"- Current version: {ctx.current.hash}",
        "- Editable scope: src/simple_agent_lab/",
        f"- Baseline runs: {len(ctx.runs)}",
        f"- Prior decisions: {len(ctx.decisions)}",
        "",
        "## Recent Decisions",
    ]
    if not ctx.decisions:
        lines.append("- none")
    for decision in ctx.decisions[-5:]:
        lines.append(f"- {decision.id}: {decision.outcome}; {decision.reason}")
    lines.extend(["", "## Baseline Runs"])
    if not ctx.runs:
        lines.append("- none")
    for run in ctx.runs[:8]:
        lines.append(f"- {run.instance_id}: reward={run.reward}")
    lines.extend(
        [
            "",
            "Inspect deeper only when these summaries are insufficient.",
            "Local evidence files live under self_evolution/.",
            "Do not edit files outside src/simple_agent_lab/ for the proposal.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_meta_agent_context(root: Path, ctx: Context, parent_hash: str) -> None:
    (root / "SELF_EVOLUTION_CONTEXT.md").write_text(
        _context_briefing(ctx, parent_hash),
        encoding="utf-8",
    )
    evidence_root = root / _META_EVIDENCE_DIR
    evidence_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        evidence_root / "current_manifest.json",
        {
            "schema": "simple-agent-lab.self-evolution-context.v1",
            "parent_version": parent_hash,
            "current_version": ctx.current.hash,
            "editable_scope": SOURCE_ROOT + "/",
            "baseline_run_count": len(ctx.runs),
            "failure_count": len(ctx.failures),
            "prior_decision_count": len(ctx.decisions),
            "current_manifest": _version_manifest(ctx.current),
        },
    )
    _write_json(
        evidence_root / "baseline_runs.json",
        {
            "schema": "simple-agent-lab.self-evolution-baseline-runs.v1",
            "runs": [_run_summary(run) for run in ctx.runs],
        },
    )
    _write_decision_jsonl(evidence_root / "prior_decisions.jsonl", ctx.decisions)
    _write_failure_artifacts(evidence_root / "failures", ctx.failures)
    (evidence_root / "README.md").write_text(
        "\n".join(
            [
                "# Self-Evolution Evidence",
                "",
                "These files are prepared for the meta-agent in this temporary repo copy.",
                "They are written into both the base and candidate trees before the agent",
                "runs, so reading them does not create a proposal diff.",
                "",
                "- current_manifest.json: parent/current version and editable scope.",
                "- baseline_runs.json: compact per-instance train-run summaries.",
                "- prior_decisions.jsonl: append-only prior evolution decisions.",
                "- failures/: result and trace excerpts for recent failing baseline runs.",
                "",
                "Only edits under src/simple_agent_lab/ can become an evolution proposal.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_decision_jsonl(path: Path, decisions: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_decision_summary(d), sort_keys=True) for d in decisions]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_failure_artifacts(root: Path, failures: Sequence[object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for run in failures[:_MAX_FAILURE_ARTIFACTS]:
        run_dir = root / _safe_artifact_name(getattr(run, "instance_id", "run"))
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "result.json", dict(getattr(run, "result", {}) or {}))
        (run_dir / "trace_excerpt.md").write_text(_trace_excerpt(run), encoding="utf-8")


def _version_manifest(version: object) -> dict[str, Any]:
    manifest = getattr(version, "manifest", None)
    if manifest is None:
        return {}
    return {
        "parent": getattr(manifest, "parent", None),
        "producer": getattr(manifest, "producer", ""),
        "evidence": list(getattr(manifest, "evidence", ())),
        "note": getattr(manifest, "note", ""),
        "created": getattr(manifest, "created", ""),
        "schema": getattr(manifest, "schema", ""),
    }


def _run_summary(run: object) -> dict[str, Any]:
    result = getattr(run, "result", {}) or {}
    summary: dict[str, Any] = {
        "instance_id": getattr(run, "instance_id", ""),
        "run_ref": getattr(run, "ref", ""),
        "ok": bool(getattr(run, "ok", False)),
        "reward": getattr(run, "reward", None),
    }
    if isinstance(result, Mapping):
        for key in ("resolved", "score", "status", "error", "message"):
            if key in result:
                summary[key] = result[key]
    return summary


def _decision_summary(decision: object) -> dict[str, Any]:
    return {
        "id": getattr(decision, "id", ""),
        "ts": getattr(decision, "ts", ""),
        "kind": getattr(decision, "kind", ""),
        "accepted": bool(getattr(decision, "accepted", False)),
        "reason": getattr(decision, "reason", ""),
        "baseline": dict(getattr(decision, "baseline", {}) or {}),
        "candidate": dict(getattr(decision, "candidate", {}) or {}),
        "deltas": dict(getattr(decision, "deltas", {}) or {}),
        "runs": dict(getattr(decision, "runs", {}) or {}),
    }


def _trace_excerpt(run: object) -> str:
    try:
        events = tuple(getattr(run, "events")())
    except Exception:
        events = ()
    lines = [
        f"# Trace Excerpt: {getattr(run, 'instance_id', '')}",
        "",
        f"- run_ref: {getattr(run, 'ref', '')}",
        f"- reward: {getattr(run, 'reward', None)}",
        "",
    ]
    if not events:
        lines.append("No trajectory events found.")
        lines.append("")
        return "\n".join(lines)
    lines.append("## Recent Events")
    for event in events[-_MAX_TRACE_EVENTS:]:
        if not isinstance(event, Mapping):
            lines.append(f"- {str(event)[:_MAX_EVENT_TEXT]}")
            continue
        label = event.get("type") or event.get("event") or event.get("kind") or "event"
        text = event.get("text") or event.get("message") or event.get("content") or ""
        lines.append(f"- {label}: {str(text)[:_MAX_EVENT_TEXT]}")
    lines.append("")
    return "\n".join(lines)


def _safe_artifact_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "run"


def _select_parent(
    ctx: Context,
    *,
    parent_selection: str,
    parent_selector: Callable[[Context, str], str] | None,
) -> str:
    if parent_selection == "current":
        return ctx.current.hash
    if parent_selector is None:
        raise ValueError(
            "non-current parent selection requires a recipe-provided parent_selector"
        )
    return parent_selector(ctx, parent_selection) or ctx.current.hash


def _overlay_source_version(
    base_tree: Path,
    files: Sequence[str],
    read: Callable[[str], str],
) -> None:
    for path in files:
        rel = Path(path)
        if not _is_under_source_root(rel):
            continue
        target = _prepare_overlay_target(base_tree, rel)
        target.write_text(read(path), encoding="utf-8")


def _prepare_overlay_target(base_tree: Path, rel: Path) -> Path:
    current = base_tree
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink():
            current.unlink()
        current.mkdir(exist_ok=True)

    target = current / rel.name
    if target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _changed_paths(base_tree: Path, changed_tree: Path) -> list[Path]:
    base_files = _file_map(base_tree)
    changed_files = _file_map(changed_tree)
    paths = sorted(set(base_files) | set(changed_files))
    changed: list[Path] = []
    for rel in paths:
        base_path = base_files.get(rel)
        changed_path = changed_files.get(rel)
        if base_path is None or changed_path is None:
            changed.append(rel)
            continue
        if base_path.is_symlink() or changed_path.is_symlink():
            if base_path.resolve(strict=False) != changed_path.resolve(strict=False):
                changed.append(rel)
            continue
        if base_path.read_bytes() != changed_path.read_bytes():
            changed.append(rel)
    return changed


def _changed_text_edits(base_tree: Path, changed_tree: Path) -> dict[str, str | None]:
    base_files = _text_file_map(base_tree)
    changed_files = _text_file_map(changed_tree)
    edits: dict[str, str | None] = {}
    for rel in sorted(set(base_files) | set(changed_files)):
        if rel not in changed_files:
            edits[rel] = None
            continue
        changed = changed_files[rel]
        if base_files.get(rel) != changed:
            edits[rel] = changed
    return edits


def _content_changing_edits(
    version: Version, edits: Mapping[str, str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    changed: dict[str, str] = {}
    unchanged: list[str] = []
    version_files = set(version.files())
    for path, value in edits.items():
        if path in version_files and version.read(path) == value:
            unchanged.append(path)
            continue
        changed[path] = value
    return changed, tuple(unchanged)


def _text_file_map(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if _skip_path(path):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def _file_map(root: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if _skip_path(path):
            continue
        if path.is_file() or path.is_symlink():
            files[path.relative_to(root)] = path
    return files


def _skip_path(path: Path) -> bool:
    return any(part in _COPY_IGNORE_NAMES for part in path.parts)


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _COPY_IGNORE_NAMES}


def _is_under_source_root(path: Path) -> bool:
    parts = path.parts
    source_parts = Path(SOURCE_ROOT).parts
    return len(parts) > len(source_parts) and parts[: len(source_parts)] == source_parts
