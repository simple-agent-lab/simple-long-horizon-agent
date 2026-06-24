from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.strategy import (
    content_changing_edits,
    parse_model_json,
)
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import Context, Proposal, Version
from simple_agent_lab.llm import LLMRequest, Provider, complete, llm_message

from recipes.ahe import ledger
from recipes.ahe.analyzer import AnalysisResult, analyze_runs


def ahe_model_strategy(
    *,
    provider: Provider,
    surface: AgentSurface,
    editable_components: Sequence[str],
    knowledge_paths: Sequence[str] = (),
    complete_fn: Callable[[LLMRequest], Any] = complete,
    analyzer_fn: Callable[..., AnalysisResult] = analyze_runs,
    max_tokens: int = 6000,
) -> Callable[[Context], Proposal | None]:
    surface_brief = ""
    if surface is not None:
        surface_brief = "\n\n" + surface.prompt_brief(components=editable_components)

    def strategy(ctx: Context) -> Proposal | None:
        run_root = ctx.workspace.parent
        round_index = len(ctx.decisions) + 1
        round_path = ledger.round_dir(run_root, round_index)

        knowledge_texts = _read_knowledge(knowledge_paths)
        analysis = analyzer_fn(
            provider,
            ctx.runs,
            ctx.current,
            ctx.decisions,
            round_path / "analysis",
            knowledge=knowledge_texts,
        )

        current_files = _read_harness_files(ctx.current)
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
            str(item)
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
            kind="ahe_harness",
        )

    return strategy


def _read_knowledge(knowledge_paths: Sequence[str]) -> tuple[str, ...]:
    texts: list[str] = []
    for raw_path in knowledge_paths:
        path = Path(raw_path)
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
        else:
            texts.append(f"missing knowledge: {path}")
    return tuple(texts)


def _read_harness_files(version: Version) -> tuple[tuple[str, str], ...]:
    files = []
    for name in version.files():
        if not name.startswith("harness/"):
            continue
        if name == "harness/provider.json":
            continue
        files.append((name, version.read(name)))
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
    decision_lines = [_format_decision(decision) for decision in decisions] or [
        "- none"
    ]
    decision_block = "\n".join(decision_lines)
    analysis_index = f"{analysis.index_path}\n{analysis.index}"
    return "\n\n".join(
        [
            f"Round index: {round_index}",
            f"Analysis overview path: {analysis.overview_path}",
            analysis.overview,
            f"Analysis index path: {analysis.index_path}",
            analysis_index,
            knowledge_block,
            "Current harness files:\n" + files_block,
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
        return f"- {decision.id}: outcome={decision.outcome}, reason={decision.reason}"
    return f"- {decision}"


def _manifest_from_payload(
    raw_manifest: object, current_hash: str, round_index: int
) -> dict[str, object]:
    if isinstance(raw_manifest, Mapping):
        manifest = dict(raw_manifest)
    else:
        manifest = {"round": round_index, "base_version": current_hash, "changes": []}
    manifest["round"] = round_index
    manifest["base_version"] = current_hash
    changes = manifest.get("changes")
    if not isinstance(changes, list):
        manifest["changes"] = []
    return manifest


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    return (str(value),)
