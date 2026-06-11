"""Lab: the user-facing surface of the evolution framework.

The integration model follows verl / slime: a researcher provides plain
functions, the framework owns the machinery. Three things to know:

    lab = Lab(workspace, rollout=...)        # what to run

    def my_reward(run_dir: Path) -> float:   # how to score one run
        ...                                   # (optional: defaults to the
                                              #  run's result.json "reward")

    def my_strategy(ctx: EpisodeContext):    # how to change the agent
        return ctx.propose(kind="lesson", edits={...}, note="...",
                           evidence=[...])    # (optional: omit and call
                                              #  lab.evolve() for the
                                              #  agent-driven mode)

    print(lab.step(my_strategy).text)        # run -> compare -> promote/reject
    print(lab.history())                     # the experiment log

Everything else — bundles, the gate, the decision log, the catalog — is the
engine room: it enforces the guarantees (immutable versions, evidence-gated
promotion, append-only history, rollback) without appearing in user code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.evolution.bundle import (
    Manifest,
    bundle_hash,
    promote,
    read_manifest,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.catalog import CatalogRow, build_catalog
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


RewardFn = Callable[[Path], float]  # one run dir in, one score out (verl-style)


@dataclass(frozen=True)
class Proposal:
    """A strategy's output: one candidate change, with provenance."""

    kind: str  # "lesson" | "playbook" | "skill" | "prompt" | "provider" | ...
    edits: Mapping[str, str]  # relative file path -> full new content
    note: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeContext:
    """What a strategy function sees: recent runs and the current version."""

    runs: tuple[CatalogRow, ...]
    current_dir: Path

    @property
    def failures(self) -> tuple[CatalogRow, ...]:
        return tuple(r for r in self.runs if r.reward is not None and r.reward <= 0.0)

    def current(self, filename: str) -> str:
        """Read one file of the current version ("" when absent)."""

        path = self.current_dir / filename
        return path.read_text() if path.exists() else ""

    def propose(
        self,
        *,
        kind: str,
        edits: Mapping[str, str],
        note: str,
        evidence: Sequence[str] = (),
    ) -> Proposal:
        return Proposal(kind=kind, edits=edits, note=note, evidence=tuple(evidence))


StrategyFn = Callable[[EpisodeContext], Proposal | None]


@dataclass(frozen=True)
class StepReport:
    """One step's outcome, as data plus a human-readable summary."""

    episode: str
    proposed: bool
    accepted: bool
    promoted_to: str  # bundle hash when promoted, "" otherwise
    text: str


def _reward_measure(reward: RewardFn) -> Measure:
    def mean_reward(runs: Sequence[Path]) -> float:
        scores = [reward(run_dir) for run_dir in runs]
        return sum(scores) / len(scores) if scores else 0.0

    return Measure("reward", mean_reward)


class Lab:
    """One experiment: a workspace, a way to run tasks, a way to score them.

    ``rollout`` is any ``(bundle_dir, run_id) -> run dirs`` callable — use
    ``simple_agent_lab.evolution.rollout.dataset_rollout`` for containerized
    eval suites, or a plain function for local/custom setups. ``reward``
    overrides scoring verl-style (one function, one run dir, one float);
    by default the standard ``result.json`` reward key is used.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        rollout: Rollout,
        reward: RewardFn | None = None,
        criterion: Criterion | None = None,
        slice_name: str = "custom",
        instances: Sequence[Mapping[str, object]] = (),
        seed: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.rollout = rollout
        self._reward = reward
        self.measures: tuple[Measure, ...] = (
            (_reward_measure(reward),) if reward else (REWARD,)
        )
        self.criterion = criterion or improve("reward")
        self.slice_ = EvalSlice(suite=slice_name, instances=tuple(instances))
        self._ensure_initial(seed or {"prompt.md": ""})

    def _ensure_initial(self, seed: Mapping[str, str]) -> None:
        try:
            resolve(self.workspace, "task")
        except FileNotFoundError:
            initial = stage_bundle(
                self.workspace,
                manifest=Manifest(level="task", producer="lab", note="initial"),
                edits=dict(seed),
            )
            promote(self.workspace, "task", initial)

    def _catalog(self) -> list[CatalogRow]:
        """Run index, scored with the custom reward when one is configured —
        so ctx.failures means the same thing the gate measures mean."""

        rows = build_catalog(self.workspace / "runs")
        if self._reward is None:
            return rows
        return [replace(r, reward=self._reward(Path(r.path))) for r in rows]

    # ---------------------------------------------------------------- #
    # the researcher loop
    # ---------------------------------------------------------------- #
    def step(self, strategy: StrategyFn) -> StepReport:
        """Run one strategy step: observe -> propose -> compare -> promote.

        The strategy only proposes; acceptance is decided by the comparison
        on the frozen slice and recorded in the experiment log, and promotion
        happens here, host-side — same guarantees as the agent-driven mode.
        """

        episode = f"step-{len(read_decisions(self.workspace)) + 1:06d}"
        current = resolve(self.workspace, "task")
        runs = self._catalog()
        if not runs:  # first contact: observe the current version once
            self.rollout(current, f"{episode}-observe")
            runs = self._catalog()

        proposal = strategy(EpisodeContext(runs=tuple(runs), current_dir=current))
        if proposal is None:
            return StepReport(episode, False, False, "", f"{episode}: no proposal")

        candidate = stage_bundle(
            self.workspace,
            base=current,
            edits=dict(proposal.edits),
            manifest=Manifest(
                level="task",
                parent=bundle_hash(current),
                producer=getattr(strategy, "__name__", "strategy"),
                evidence=proposal.evidence,
                note=proposal.note,
            ),
        )
        result = gate(
            self.workspace,
            baseline=current,
            candidate=candidate,
            slice_=self.slice_,
            rollout=self.rollout,
            measures=self.measures,
            criterion=self.criterion,
            episode=episode,
            kind=proposal.kind,
        )
        promoted = ""
        if result.judgment.accepted:
            promote(self.workspace, "task", candidate)
            promoted = bundle_hash(candidate)
        verdict = "ACCEPTED, promoted" if promoted else "rejected"
        return StepReport(
            episode,
            True,
            result.judgment.accepted,
            promoted,
            f"{episode}: {proposal.kind} {verdict} — {result.judgment.reason}",
        )

    def evolve(
        self, provider: Provider, *, episodes: int = 1, max_turns: int = 12
    ) -> list[str]:
        """Agent-driven mode: no strategy function, the evolution agent
        diagnoses and chooses the intervention itself. Returns one summary
        line per episode."""

        # Imported here so the strategy-function path never needs an LLM stack.
        from simple_agent_lab.evolution.agent import EvolutionConfig, run_episode

        config = EvolutionConfig(
            workspace=self.workspace,
            rollout=self.rollout,
            slice_=self.slice_,
            measures=self.measures,
            criterion=self.criterion,
        )
        try:
            resolve(self.workspace, "meta")
        except FileNotFoundError:  # the agent needs its own (meta) bundle
            promote(self.workspace, "meta", resolve(self.workspace, "task"))
        lines = []
        for _ in range(episodes):
            report = run_episode(provider, config, max_turns=max_turns)
            lines.append(
                f"{report.episode}: decisions={list(report.decisions)} "
                f"promoted={list(report.promoted)}"
            )
        return lines

    def history(self, *, limit: int | None = None) -> str:
        """The experiment log, oldest first — every comparison ever made."""

        rows = read_decisions(self.workspace, limit=limit)
        if not rows:
            return "no decisions yet"
        return "\n".join(f"{d.id} [{d.kind}] {d.decision}: {d.reason}" for d in rows)

    def rollback(self) -> str:
        """Move the current version back to its parent. Nothing is deleted."""

        current = resolve(self.workspace, "task")
        parent_hash = read_manifest(current).parent
        if not parent_hash:
            return "already at the initial version"
        parent_dir = self.workspace / "bundles" / parent_hash
        promote(self.workspace, "task", parent_dir)
        return f"rolled back: task -> {parent_hash}"
