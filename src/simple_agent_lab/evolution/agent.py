"""The evolution agent: an ordinary Agent whose tools are the substrate.

The agent owns the open decisions (which failures matter, which intervention
kind fits, what the candidate says); the substrate owns the guarantees. The
tool layer is where authority is enforced: the agent can only stage
candidates and only promote through the gate — pointers, the decision log,
and promoted bundles are out of its reach.

A "meta" candidate (kind="meta") edits the evolution agent's own bundle; it
takes effect at the next episode, never mid-flight (spec invariant 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.evolution.bundle import (
    Bundle,
    Manifest,
    promote,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.catalog import Run, build_catalog, format_rows
from simple_agent_lab.evolution.decisions import read_decisions
from simple_agent_lab.evolution.gate import (
    Criterion,
    EvalSlice,
    Measure,
    REWARD,
    Rollout,
    gate,
    improve,
)


TASK_KINDS = ("lesson", "playbook", "skill", "prompt", "context_policy", "provider")
META_KIND = "meta"

DEFAULT_PROMPT = """\
You are the evolution agent for an agent system. Each episode: inspect recent
runs (query_runs), localize failure modes (read_trace), check what was tried
before (read_decisions), then draft ONE candidate change (write_candidate) and
submit it to the gate (run_gate). A candidate is a small file edit: a lesson,
a playbook bullet, a skill, a prompt change, or — when the way you yourself
work is the bottleneck — a "meta" edit to your own bundle. By default you
build on the current bundle, but you may branch from ANY archived bundle via
base=<hash> — a rejected candidate from an earlier episode can be the right
starting point (read_decisions shows what exists). Prefer the cheapest
intervention that addresses the diagnosed cause. Interpret the gate verdict in
your final message: what you tried, why, and what you would try next.
"""


@dataclass
class EvolutionConfig:
    """Everything one episode needs. Plain data; build it in a script."""

    workspace: Path
    rollout: Rollout
    slice_: EvalSlice
    measures: Sequence[Measure] = (REWARD,)
    criterion: Criterion = field(default_factory=lambda: improve("reward"))
    max_gates_per_episode: int = 2


def make_evolution_tools(config: EvolutionConfig, *, episode: str) -> list[AgentTool]:
    workspace = config.workspace
    gates_used = [0]  # closure-mutable budget counter

    def query_runs(_call_id, args, _abort, _on_update) -> ToolResult:
        rows = build_catalog(workspace / "runs")
        return text_result(
            format_rows(
                rows,
                failed_only=bool(args.get("failed_only", False)),
                limit=int(args.get("limit", 20)),
            )
        )

    def read_trace(_call_id, args, _abort, _on_update) -> ToolResult:
        raw = str(args.get("run", ""))
        path = Path(raw)
        if not path.is_absolute():  # the normal case: a workspace-relative ref
            path = workspace / "runs" / raw
        run = Run(path)
        if not run.ok:
            return text_result(f"no out/result.json under {run.dir}", is_error=True)
        parts = [f"result: {json.dumps(run.result)}"]
        events = run.events()
        if events:
            tail = [
                f"- {e.get('kind', '?')}: {json.dumps(e, default=str)[:200]}"
                for e in events[-int(args.get("tail", 10)) :]
            ]
            parts.append("trace tail:\n" + "\n".join(tail))
        return text_result("\n".join(parts))

    def write_candidate(_call_id, args, _abort, _on_update) -> ToolResult:
        kind = str(args.get("kind", ""))
        if kind not in (*TASK_KINDS, META_KIND):
            return text_result(
                f"unknown kind {kind!r}; expected one of {(*TASK_KINDS, META_KIND)}",
                is_error=True,
            )
        pointer = "meta" if kind == META_KIND else "task"
        base_hash = str(args.get("base", "") or "")
        if base_hash:  # branch from a stepping stone instead of the tip
            base_dir = workspace / "bundles" / base_hash
            if not base_dir.is_dir():
                return text_result(
                    f"unknown base bundle {base_hash!r}", is_error=True
                )
            base = Bundle(base_dir)
        else:
            base = resolve(workspace, pointer)
        candidate = stage_bundle(
            workspace,
            base=base,
            edits={
                # JSON null is the deletion tombstone (retire a file).
                str(k): (None if v is None else str(v))
                for k, v in dict(args.get("edits", {})).items()
            },
            manifest=Manifest(
                level=pointer,
                parent=base.hash,
                producer=f"evolution-agent/{episode}",
                evidence=tuple(args.get("evidence", ())),
                note=str(args.get("note", "")),
            ),
        )
        return text_result(f"staged candidate {candidate.hash} (kind={kind})")

    def run_gate(_call_id, args, _abort, _on_update) -> ToolResult:
        if gates_used[0] >= config.max_gates_per_episode:
            return text_result(
                f"gate budget exhausted ({config.max_gates_per_episode} per episode)",
                is_error=True,
            )
        candidate_dir = workspace / "bundles" / str(args.get("candidate", ""))
        if not candidate_dir.is_dir():
            return text_result(
                f"no staged bundle {args.get('candidate')!r}", is_error=True
            )
        pointer = "meta" if str(args.get("kind", "")) == META_KIND else "task"
        gates_used[0] += 1
        result = gate(
            workspace,
            baseline=resolve(workspace, pointer),
            candidate=Bundle(candidate_dir),
            slice_=config.slice_,
            rollout=config.rollout,
            measures=config.measures,
            criterion=config.criterion,
            episode=episode,
            kind=str(args.get("kind", "")),
        )
        verdict = "ACCEPTED" if result.judgment.accepted else "REJECTED"
        base_agg = {k: f.value for k, f in result.baseline.items()}
        cand_agg = {k: f.value for k, f in result.candidate.items()}
        return text_result(
            f"{result.decision_id}: {verdict} — {result.judgment.reason}\n"
            f"baseline={base_agg} candidate={cand_agg}"
        )

    def read_decisions_tool(_call_id, args, _abort, _on_update) -> ToolResult:
        rows = read_decisions(workspace, limit=int(args.get("limit", 10)))
        if not rows:
            return text_result("decision log is empty")
        lines = [f"{d.id} [{d.level}/{d.kind}] {d.decision}: {d.reason}" for d in rows]
        return text_result("\n".join(lines))

    return [
        AgentTool(
            name="query_runs",
            description="Index recent runs: instance, bundle, reward, run path.",
            parameters={
                "type": "object",
                "properties": {
                    "failed_only": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
            },
            execute=query_runs,
        ),
        AgentTool(
            name="read_trace",
            description="Read one run's result and trace tail to localize a failure.",
            parameters={
                "type": "object",
                "properties": {
                    "run": {
                        "type": "string",
                        "description": "run ref (run_id/instance_id) from query_runs",
                    },
                    "tail": {"type": "integer"},
                },
                "required": ["run"],
            },
            execute=read_trace,
        ),
        AgentTool(
            name="write_candidate",
            description=(
                "Stage a candidate bundle: file edits on top of the current "
                "bundle, or on top of any archived bundle via `base` "
                "(rejected candidates are valid stepping stones). kind=meta "
                "edits your own bundle (takes effect next episode). Staging "
                "only — promotion goes through the gate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list((*TASK_KINDS, META_KIND))},
                    "edits": {
                        "type": "object",
                        "description": "relative file path -> full new content; "
                        "null deletes the file (retire a skill/lesson)",
                    },
                    "note": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "run refs (run_id/instance_id) or trace "
                        "ids — never absolute paths (the log outlives machines)",
                    },
                    "base": {
                        "type": "string",
                        "description": "bundle hash to branch from "
                        "(default: the current bundle)",
                    },
                },
                "required": ["kind", "edits", "note"],
            },
            execute=write_candidate,
        ),
        AgentTool(
            name="run_gate",
            description="A/B the staged candidate against the current bundle on the frozen slice.",
            parameters={
                "type": "object",
                "properties": {
                    "candidate": {
                        "type": "string",
                        "description": "staged bundle hash",
                    },
                    "kind": {"type": "string"},
                },
                "required": ["candidate", "kind"],
            },
            execute=run_gate,
        ),
        AgentTool(
            name="read_decisions",
            description="Recent gate decisions: what was tried, what worked, why.",
            parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
            execute=read_decisions_tool,
        ),
    ]


def make_evolution_agent(
    provider: Provider, config: EvolutionConfig, *, episode: str
) -> Agent:
    """The evolution agent, loading its own (meta) bundle for its prompt."""

    meta_bundle = resolve(config.workspace, "meta")
    system_prompt = meta_bundle.read("prompt.md") or DEFAULT_PROMPT
    return make_llm_agent(
        name="evolution",
        provider=provider,
        system_prompt=system_prompt,
        tools=make_evolution_tools(config, episode=episode),
    )


@dataclass(frozen=True)
class EpisodeReport:
    episode: str
    decisions: tuple[str, ...]  # decision ids made this episode
    promoted: tuple[str, ...]  # bundle hashes promoted this episode


def run_episode(
    provider: Provider, config: EvolutionConfig, *, max_turns: int = 12
) -> EpisodeReport:
    """One evolution episode: run the agent, then promote what the gate accepted.

    Promotion is host-side and evidence-driven — the agent never moves
    pointers; this loop reads the decision log and moves them for it. The
    episode itself is saved as a trace under ``episodes/``: why a candidate
    was proposed and how the verdict was read is auditable like any run.
    """

    episode = f"ep-{len(read_decisions(config.workspace)) + 1:06d}"
    agent = make_evolution_agent(provider, config, episode=episode)
    state, events = agent.run(
        f"Evolution episode {episode}. Eval slice: {config.slice_.describe()}. "
        f"Diagnose, draft one candidate, gate it, and report.",
        max_turns=max_turns,
    )
    for _ in events:  # drive the loop; state records everything
        pass
    _save_episode_trace(config.workspace, episode, state)

    promoted = []
    for decision in read_decisions(config.workspace, episode=episode):
        if decision.decision != "accepted":
            continue
        bundle = Bundle(config.workspace / "bundles" / decision.candidate["bundle"])
        promote(config.workspace, decision.level, bundle)
        promoted.append(decision.candidate["bundle"])
    made = tuple(d.id for d in read_decisions(config.workspace, episode=episode))
    return EpisodeReport(episode=episode, decisions=made, promoted=tuple(promoted))


def _save_episode_trace(workspace: Path, episode: str, state) -> None:
    from simple_agent_lab.trace.run_trace import run_trace_from_state, trace_record

    trace = run_trace_from_state(
        state=state, trace_id=episode, producer="evolution-agent"
    )
    episodes = workspace / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    (episodes / f"{episode}.trajectory.json").write_text(
        json.dumps(trace_record(trace), default=str)
    )
