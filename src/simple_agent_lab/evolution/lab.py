"""Lab: the user-facing surface of the evolution framework.

The integration model follows verl / slime: a researcher provides plain
functions, the framework owns the machinery. Three things to know:

    lab = Lab(workspace, rollout=...)        # what to run

    def my_reward(run: Run) -> float:        # how to score one run —
        return run.result["resolved"]         # run.result / run.reward /
                                              # run.events() / run.dir
                                              # (optional: defaults to the
                                              #  result.json "reward" key)

    def my_strategy(ctx: EpisodeContext):    # how to change the agent
        return ctx.propose(kind="lesson", edits={...}, note="...",
                           evidence=[...])    # (optional: omit and call
                                              #  lab.evolve() for the
                                              #  agent-driven mode)

    print(lab.step(my_strategy).text)        # run -> compare -> promote/reject
    print(lab.history())                     # the experiment log

Extension points take typed views (``Run``, ``Bundle``), never raw paths:
the directory contract is discoverable from the type, and ``.dir`` stays
the escape hatch. Everything else — the gate, the decision log, the
catalog — is the engine room: it enforces the guarantees (immutable
versions, evidence-gated promotion, append-only history, rollback) without
appearing in user code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.evolution.bundle import (
    Bundle,
    Manifest,
    promote,
    resolve,
    stage_bundle,
)
from simple_agent_lab.evolution.catalog import Run, build_catalog
from simple_agent_lab.evolution.decisions import Decision, read_decisions
from simple_agent_lab.evolution.gate import (
    COST_TOKENS,
    Criterion,
    EvalSlice,
    Measure,
    REWARD,
    Rollout,
    gate,
    improve,
)


RewardFn = Callable[[Run], float]  # one run in, one score out (verl-style)


def default_reward(run: Run) -> float:
    """The standard scoring: the result.json reward key; a crashed run is 0."""

    return run.reward if run.reward is not None else 0.0


@dataclass(frozen=True)
class Proposal:
    """A strategy's output: one candidate change, with provenance.

    ``base`` branches the candidate off any archived bundle — rejected ones
    included (the DGM stepping-stone move) — instead of the current version;
    the gate baseline stays the current version either way. ``criterion``
    lets one proposal carry its own success definition (e.g. an efficiency
    objective) without reconfiguring the Lab.
    """

    kind: str  # "lesson" | "playbook" | "skill" | "prompt" | "provider" | ...
    edits: Mapping[str, str | bytes | None]  # path -> content; None retires it
    note: str
    evidence: tuple[str, ...] = ()
    base: str = ""  # bundle hash to branch from; "" = the current version
    criterion: Criterion | None = None  # None = the Lab's criterion


@dataclass(frozen=True)
class EpisodeContext:
    """What a strategy function sees: recent runs, the current version, and
    the experiment history (every bundle ever staged is branchable).

    All typed views: ``runs`` are ``Run`` (result/reward/events/dir),
    ``current`` is a ``Bundle`` (read/manifest/dir). ``reward`` is the same
    scoring definition the gate measures, so ``failures`` and the gate
    agree by construction."""

    runs: tuple[Run, ...]
    current: Bundle
    workspace: Path
    decisions: tuple[Decision, ...] = ()
    reward: RewardFn = default_reward

    @property
    def failures(self) -> tuple[Run, ...]:
        """Runs scoring <= 0 (verifier-style). For a continuous reward,
        filter ``self.runs`` with your own cut — it is one line."""

        return tuple(r for r in self.runs if self.reward(r) <= 0.0)

    def bundle(self, hash_: str) -> Bundle:
        """Any archived bundle — rejected candidates included; pair with
        ``propose(base=...)`` to branch from a stepping stone."""

        return Bundle(self.workspace / "bundles" / hash_)

    def propose(
        self,
        *,
        kind: str,
        edits: Mapping[str, str | bytes | None],
        note: str,
        evidence: Sequence[str] = (),
        base: str = "",
        criterion: Criterion | None = None,
    ) -> Proposal:
        return Proposal(
            kind=kind,
            edits=edits,
            note=note,
            evidence=tuple(evidence),
            base=base,
            criterion=criterion,
        )


StrategyFn = Callable[[EpisodeContext], Proposal | None]


@dataclass(frozen=True)
class StepReport:
    """One step's outcome, as data plus a human-readable summary."""

    episode: str
    proposed: bool
    accepted: bool
    candidate: str  # staged bundle hash ("" when nothing was proposed)
    promoted_to: str  # bundle hash when promoted, "" otherwise
    text: str


class Lab:
    """One experiment: a workspace, a way to run tasks, a way to score them.

    ``rollout`` is any ``(bundle_dir, run_id) -> run dirs`` callable — use
    ``simple_agent_lab.evolution.rollout.dataset_rollout`` for containerized
    eval suites, or a plain function for local/custom setups. ``reward``
    overrides scoring verl-style (one function, one run dir, one float);
    by default the standard ``result.json`` reward key is used.

    The defaults encode one search policy (single lineage, climb on reward,
    promote on acceptance) — every part of it is swappable: a proposal may
    branch from any archived bundle (``base=``) and carry its own
    ``criterion``; ``auto_promote=False`` adds a confirmation tier where
    gate-accepted candidates wait for an explicit ``lab.promote(hash)``.
    Criterion dimension names resolve automatically — "reward" means this
    Lab's reward, built-ins (cost_tokens) by name, anything else the
    same-named result.json field — so referencing a dimension registers it.
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
        auto_promote: bool = True,
        runs_root: Path | None = None,  # where rollouts land; default ws/runs
    ) -> None:
        self.workspace = Path(workspace)
        self.rollout = rollout
        self._reward = reward or default_reward
        # A custom reward slots in directly as the gate's reward measure.
        self._reward_measure = Measure("reward", per_run=reward) if reward else REWARD
        self.criterion = criterion or improve("reward")
        self.slice_ = EvalSlice(suite=slice_name, instances=tuple(instances))
        self.auto_promote = auto_promote
        self.runs_root = Path(runs_root) if runs_root else self.workspace / "runs"
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

    def _catalog(self) -> list[Run]:
        return build_catalog(self.runs_root)

    def _measures_for(self, criterion: Criterion) -> tuple[Measure, ...]:
        """Referencing a dimension IS registering it: resolve the names the
        criterion reads — "reward" -> this Lab's reward definition,
        built-ins (cost_tokens) by name, anything else -> the same-named
        numeric result.json field. No measure plumbing in user code."""

        return tuple(
            self._resolve_measure(name) for name in (criterion.requires or ("reward",))
        )

    def _resolve_measure(self, name: str) -> Measure:
        if name == "reward":
            return self._reward_measure
        if name == "cost_tokens":
            return COST_TOKENS

        def from_result(run: Run, _name: str = name) -> float:
            if _name not in run.result:
                raise ValueError(
                    f"criterion reads dimension {_name!r} but "
                    f"{run.dir}/out/result.json has no such field"
                )
            return float(run.result[_name])

        return Measure(name, per_run=from_result)

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
            self.rollout(current, self.slice_, f"{episode}-observe")
            runs = self._catalog()

        proposal = strategy(
            EpisodeContext(
                runs=tuple(runs),
                current=current,
                workspace=self.workspace,
                decisions=tuple(read_decisions(self.workspace)),
                reward=self._reward,
            )
        )
        if proposal is None:
            return StepReport(episode, False, False, "", "", f"{episode}: no proposal")

        base = current
        if proposal.base:  # branch from a stepping stone instead of the tip
            base_dir = self.workspace / "bundles" / proposal.base
            if not base_dir.is_dir():
                return StepReport(
                    episode,
                    True,
                    False,
                    "",
                    "",
                    f"{episode}: unknown base bundle {proposal.base!r}",
                )
            base = Bundle(base_dir)
        candidate = stage_bundle(
            self.workspace,
            base=base,
            edits=dict(proposal.edits),
            manifest=Manifest(
                level="task",
                parent=base.hash,
                producer=getattr(strategy, "__name__", "strategy"),
                evidence=proposal.evidence,
                note=proposal.note,
            ),
        )
        criterion = proposal.criterion or self.criterion
        result = gate(
            self.workspace,
            baseline=current,  # always compared against what runs now
            candidate=candidate,
            slice_=self.slice_,
            rollout=self.rollout,
            measures=self._measures_for(criterion),
            criterion=criterion,
            runs_root=self.runs_root,
            episode=episode,
            kind=proposal.kind,
        )
        candidate_hash = candidate.hash
        promoted = ""
        if result.judgment.accepted and self.auto_promote:
            promote(self.workspace, "task", candidate)
            promoted = candidate_hash
        if promoted:
            verdict = "ACCEPTED, promoted"
        elif result.judgment.accepted:
            verdict = f"ACCEPTED, awaiting promote({candidate_hash!r})"
        else:
            verdict = "rejected"
        return StepReport(
            episode,
            True,
            result.judgment.accepted,
            candidate_hash,
            promoted,
            f"{episode}: {proposal.kind} {verdict} — {result.judgment.reason}",
        )

    def promote(self, bundle: str) -> str:
        """Promote a gate-accepted candidate — the confirmation tier of
        ``step()`` when ``auto_promote=False``.

        Evidence stays mandatory in both modes: a hash with no accepted
        decision in the log is refused, so this cannot bypass the gate.
        """

        accepted = {
            d.candidate.get("bundle")
            for d in read_decisions(self.workspace, level="task")
            if d.decision == "accepted"
        }
        if bundle not in accepted:
            return f"refused: no accepted decision for {bundle!r} — run step() first"
        promote(self.workspace, "task", Bundle(self.workspace / "bundles" / bundle))
        return f"promoted: task -> {bundle}"

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
            measures=self._measures_for(self.criterion),
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

        parent_hash = resolve(self.workspace, "task").parent
        if not parent_hash:
            return "already at the initial version"
        promote(self.workspace, "task", Bundle(self.workspace / "bundles" / parent_hash))
        return f"rolled back: task -> {parent_hash}"
