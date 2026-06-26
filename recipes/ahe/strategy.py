from __future__ import annotations

import json
import shutil
import tempfile
import fnmatch
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.evolution.components.agentic import (
    RunnableAgent,
    consume_agent_run,
    materialize_version_files,
    surface_proposal_from_trees,
)
from simple_agent_lab.evolution.components.strategy import (
    content_changing_edits,
    parse_model_json,
)
from simple_agent_lab.evolution.kernel.loop import score
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import Context, Proposal, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

from recipes.ahe import ledger
from recipes.ahe.analyzer import AnalysisResult, analyze_runs

MAX_KNOWLEDGE_SNIPPETS = 6
MAX_KNOWLEDGE_CHARS = 320
MAX_HARNESS_FILES = 8
MAX_HARNESS_FILE_CHARS = 900
MAX_ANALYSIS_OVERVIEW_CHARS = 1200
MAX_ANALYSIS_INDEX_CHARS = 1800
MAX_PRIOR_DECISIONS = 5
MAX_DECISION_CHARS = 240
MAX_EVIDENCE_CHARS = 320

AHE_EVOLVE_TASK = """Improve this AHE source-tree surface.

Read runs/iteration_001/input/analysis/overview.md and index.json first, then
make one focused edit to the selected AgentSurface. Write change_manifest.json
at the workspace root describing the component-level change. Do not edit files
outside the editable surface except change_manifest.json.
"""


def ahe_model_strategy(
    *,
    provider: Provider,
    surface: AgentSurface,
    editable_components: Sequence[str],
    knowledge_paths: Sequence[str] = (),
    complete_fn: Callable[[LLMRequest], Any] = complete,
    analyzer_fn: Callable[..., AnalysisResult] = analyze_runs,
    max_tokens: int = 6000,
    repo_root: object | None = None,
) -> Callable[[Context], Proposal | None]:
    _ = repo_root
    surface_brief = ""
    if surface is not None:
        surface_brief = "\n\n" + surface.prompt_brief(components=editable_components)

    def strategy(ctx: Context) -> Proposal | None:
        run_root = ctx.workspace.parent
        round_index = _next_round_index(run_root, preferred=len(ctx.decisions) + 1)
        round_path = ledger.round_dir(run_root, round_index)

        knowledge_texts = _read_knowledge(knowledge_paths)
        run_scores = score(ctx.runs, ctx.reward)
        analysis = analyzer_fn(
            provider,
            ctx.runs,
            ctx.current,
            ctx.decisions,
            round_path / "analysis",
            knowledge=knowledge_texts,
            run_scores=run_scores,
        )

        current_files = _read_surface_files(ctx.current, surface, editable_components)
        user_prompt = _build_user_prompt(
            analysis=analysis,
            round_index=round_index,
            knowledge_texts=knowledge_texts,
            current_files=current_files,
            decisions=ctx.decisions,
        )
        system_prompt = (
            "You are an AHE meta-agent.\n"
            "Be evidence-driven and choose the smallest component-level edit that best fits the analysis.\n"
            "Return only JSON with keys: note, evidence, manifest, edits.\n"
            "The manifest should describe the change at the component level.\n"
            "The edits object must map file paths to full replacement contents.\n"
            + surface_brief
        )

        response = complete_fn(
            LLMRequest(
                provider=provider,
                messages=[llm_message("user", user_prompt)],
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        )
        try:
            payload = parse_model_json(response.text)
        except (ValueError, TypeError):
            return None

        note = str(payload.get("note", ""))
        evidence = tuple(
            _clip_text(item, MAX_EVIDENCE_CHARS)
            for item in _string_items(payload.get("evidence", ()))
            if str(item)
        )

        manifest = _manifest_from_payload(
            payload.get("manifest"), ctx.current.hash, round_index
        )
        ledger.write_json(round_path / "change_manifest.json", manifest)
        manifest_marker = (round_path / "change_manifest.json").relative_to(run_root)
        evidence += (f"manifest:{manifest_marker.as_posix()}",)

        raw_edits = payload.get("edits", {})
        if isinstance(raw_edits, Mapping):
            validated = surface.validate_edits(
                raw_edits, components=editable_components
            )
            edits = validated.edits
            rejected = validated.rejected
        else:
            edits = {}
            rejected = ()

        evidence += tuple(f"discarded-disallowed-path:{path}" for path in rejected)
        edits, unchanged = content_changing_edits(ctx.current, edits)
        evidence += tuple(f"discarded-unchanged-path:{path}" for path in unchanged)
        if not edits:
            return None

        return Proposal(
            base="",
            edits=edits,
            note=note,
            evidence=evidence,
            kind="ahe_source_tree",
        )

    return strategy


def ahe_agent_strategy(
    *,
    provider: Provider,
    surface: AgentSurface,
    editable_components: Sequence[str],
    knowledge_paths: Sequence[str] = (),
    agent_builder: Callable[..., RunnableAgent] | None = None,
    analyzer_fn: Callable[..., AnalysisResult] = analyze_runs,
    max_turns: int = 20,
    task: str = AHE_EVOLVE_TASK,
    repo_root: object | None = None,
) -> Callable[[Context], Proposal | None]:
    """Run an AHE-style evolve agent over a materialized harness workspace."""

    _ = repo_root
    build_agent = agent_builder or make_bash_agent

    def strategy(ctx: Context) -> Proposal | None:
        run_root = ctx.workspace.parent
        round_index = _next_round_index(run_root, preferred=len(ctx.decisions) + 1)
        round_path = ledger.round_dir(run_root, round_index)
        analysis_dir = round_path / "analysis"
        knowledge_texts = _read_knowledge(knowledge_paths)
        run_scores = score(ctx.runs, ctx.reward)
        analysis = analyzer_fn(
            provider,
            ctx.runs,
            ctx.current,
            ctx.decisions,
            analysis_dir,
            knowledge=knowledge_texts,
            run_scores=run_scores,
        )

        with tempfile.TemporaryDirectory(prefix="sal-ahe-agentic-") as tmp:
            root = Path(tmp)
            base_tree = root / "base"
            candidate = root / "candidate"
            materialize_version_files(ctx.current, base_tree)
            shutil.copytree(base_tree, candidate)
            _stage_ahe_evidence_workspace(
                candidate,
                ctx=ctx,
                analysis=analysis,
                knowledge=knowledge_texts,
                round_index=round_index,
            )
            agent = build_agent(
                provider=provider,
                cwd=candidate,
                name="ahe_evolve_agent",
                system_prompt=_ahe_evolve_system_prompt(surface, editable_components),
            )
            consume_agent_run(agent, task, max_turns=max_turns)
            manifest = _manifest_from_payload(
                _read_agent_manifest(candidate / "change_manifest.json"),
                ctx.current.hash,
                round_index,
            )
            ledger.write_json(round_path / "change_manifest.json", manifest)
            evidence = (
                "ahe-evolve-agent-ran",
                f"manifest:{(round_path / 'change_manifest.json').relative_to(run_root).as_posix()}",
            )
            proposal = surface_proposal_from_trees(
                base_version=ctx.current,
                base_tree=base_tree,
                changed_tree=candidate,
                surface=surface,
                editable_components=editable_components,
                base_hash=ctx.current.hash,
                note="AHE evolve agent harness edit",
                evidence=evidence,
                kind="ahe_source_tree",
            )
            if not proposal.edits:
                return None
            return proposal

    return strategy


def _stage_ahe_evidence_workspace(
    workspace: Path,
    *,
    ctx: Context,
    analysis: AnalysisResult,
    knowledge: Sequence[str],
    round_index: int,
) -> None:
    analysis_input = workspace / "runs" / "iteration_001" / "input" / "analysis"
    analysis_input.mkdir(parents=True, exist_ok=True)
    if analysis.overview_path.is_file():
        shutil.copy2(analysis.overview_path, analysis_input / "overview.md")
    if analysis.index_path.is_file():
        shutil.copy2(analysis.index_path, analysis_input / "index.json")
    context_lines = [
        "# AHE Evolve Context",
        "",
        f"- Round: {round_index}",
        f"- Current version: {ctx.current.hash}",
        f"- Baseline runs: {len(ctx.runs)}",
        f"- Prior decisions: {len(ctx.decisions)}",
        "",
        "Only edits inside the selected AgentSurface become candidate changes.",
    ]
    (workspace / "AHE_CONTEXT.md").write_text(
        "\n".join(context_lines) + "\n", encoding="utf-8"
    )
    if knowledge:
        knowledge_dir = workspace / "knowledge"
        knowledge_dir.mkdir(exist_ok=True)
        for index, text in enumerate(knowledge, start=1):
            (knowledge_dir / f"knowledge_{index:02d}.md").write_text(
                text, encoding="utf-8"
            )


def _ahe_evolve_system_prompt(
    surface: AgentSurface, editable_components: Sequence[str]
) -> str:
    return (
        "You are an AHE Evolve Agent. Use bash to inspect the evidence workspace, "
        "edit the selected AgentSurface, and write change_manifest.json.\n\n"
        + surface.prompt_brief(components=editable_components)
    )


def _read_agent_manifest(path: Path) -> object:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _next_round_index(run_root: Path, *, preferred: int) -> int:
    round_index = max(1, preferred)
    while (ledger.ahe_root(run_root) / "rounds" / f"round_{round_index:03d}").exists():
        round_index += 1
    return round_index


def _read_knowledge(knowledge_paths: Sequence[str]) -> tuple[str, ...]:
    texts: list[str] = []
    for raw_path in sorted(str(path) for path in knowledge_paths):
        path = Path(raw_path)
        if path.is_file():
            texts.append(
                _clip_text(path.read_text(encoding="utf-8"), MAX_KNOWLEDGE_CHARS)
            )
        else:
            texts.append(_clip_text(f"missing knowledge: {path}", MAX_KNOWLEDGE_CHARS))
    return tuple(texts[:MAX_KNOWLEDGE_SNIPPETS])


def _read_surface_files(
    version: Version,
    surface: AgentSurface,
    editable_components: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    selected = tuple(editable_components) or tuple(
        component.id for component in surface.components
    )
    allowed_patterns = tuple(
        pattern
        for component_id in selected
        for pattern in surface.component(component_id).paths
    )
    files = []
    for name in sorted(version.files()):
        if name == "provider.json":
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in surface.excluded_paths):
            continue
        if not any(fnmatch.fnmatch(name, pattern) for pattern in allowed_patterns):
            continue
        files.append((name, _clip_text(version.read(name), MAX_HARNESS_FILE_CHARS)))
    if len(files) > MAX_HARNESS_FILES:
        files = files[:MAX_HARNESS_FILES]
    return tuple(files)


def _build_user_prompt(
    *,
    analysis: AnalysisResult,
    round_index: int,
    knowledge_texts: Sequence[str],
    current_files: Sequence[tuple[str, str]],
    decisions: Sequence[object],
) -> str:
    knowledge_block = _section("Knowledge", knowledge_texts)
    files_block = (
        "\n\n".join(f"### {path}\n{text}" for path, text in current_files) or "- none"
    )
    decision_lines = _format_decision_lines(decisions)
    decision_block = "\n".join(decision_lines)
    overview_text = _clip_text(analysis.overview, MAX_ANALYSIS_OVERVIEW_CHARS)
    analysis_index = (
        f"{analysis.index_path}\n"
        f"{_clip_text(_deterministic_json(analysis.index), MAX_ANALYSIS_INDEX_CHARS)}"
    )
    return "\n\n".join(
        [
            f"Round index: {round_index}",
            f"Analysis overview path: {analysis.overview_path}",
            overview_text,
            f"Analysis index path: {analysis.index_path}",
            analysis_index,
            knowledge_block,
            "Current editable files:\n" + files_block,
            "Prior decisions:\n" + decision_block,
            "Return JSON with keys note, evidence, manifest, edits.",
        ]
    )


def _section(title: str, items: Sequence[str]) -> str:
    if not items:
        return f"{title}: none."
    lines = [f"{title}:"]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _format_decision(decision: object) -> str:
    if (
        hasattr(decision, "id")
        and hasattr(decision, "outcome")
        and hasattr(decision, "reason")
    ):
        return _clip_text(
            f"- {decision.id}: outcome={decision.outcome}, reason={decision.reason}",
            MAX_DECISION_CHARS,
        )
    return _clip_text(f"- {decision}", MAX_DECISION_CHARS)


def _manifest_from_payload(
    raw_manifest: object, current_hash: str, round_index: int
) -> dict[str, object]:
    if isinstance(raw_manifest, Mapping):
        manifest = dict(raw_manifest)
    else:
        manifest = {}
    manifest["round"] = round_index
    manifest["base_version"] = current_hash
    manifest["changes"] = _normalize_manifest_changes(manifest.get("changes"))
    return manifest


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    return (str(value),)


def _format_decision_lines(decisions: Sequence[object]) -> list[str]:
    if not decisions:
        return ["- none"]
    selected = list(decisions[-MAX_PRIOR_DECISIONS:])
    lines = [_format_decision(decision) for decision in selected]
    if len(decisions) > MAX_PRIOR_DECISIONS:
        lines.insert(
            0, f"- ... {len(decisions) - MAX_PRIOR_DECISIONS} earlier decisions omitted"
        )
    return lines


def _normalize_manifest_changes(raw_changes: object) -> list[dict[str, object]]:
    if not isinstance(raw_changes, list):
        return []
    changes: list[dict[str, object]] = []
    for index, raw_change in enumerate(raw_changes, start=1):
        if not isinstance(raw_change, Mapping):
            continue
        changes.append(
            _normalize_manifest_change(cast("Mapping[str, object]", raw_change), index)
        )
    return changes


def _normalize_manifest_change(
    raw_change: Mapping[str, object], index: int
) -> dict[str, object]:
    change_id = _string_field(raw_change.get("id")) or f"chg-{index}"
    return {
        "id": change_id,
        "type": _string_field(raw_change.get("type")),
        "component": _component_field(raw_change.get("component")),
        "files": _string_list(raw_change.get("files")),
        "failure_pattern": _string_field(raw_change.get("failure_pattern")),
        "root_cause": _string_field(raw_change.get("root_cause")),
        "targeted_fix": _string_field(raw_change.get("targeted_fix")),
        "predicted_fixes": _string_list(raw_change.get("predicted_fixes")),
        "risk_tasks": _string_list(raw_change.get("risk_tasks")),
        "why_this_component": _string_field(raw_change.get("why_this_component")),
    }


def _string_field(value: object) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""


def _component_field(value: object) -> str:
    if isinstance(value, str):
        return value
    return "unknown"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _clip_text(text: object, limit: int) -> str:
    value = str(text)
    if limit < 0 or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _deterministic_json(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
