"""Experiment: the slim wirer. Holds the four components + workspace + slice and
exposes step / run / history / rollback. No policy lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.criterion import Criterion, improve
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.kernel import log, loop, store
from simple_agent_lab.evolution.types import (
    Context,
    Decision,
    Manifest,
    Proposal,
    RewardFn,
    Slice,
    Version,
)

Strategy = Callable[[Context], "Proposal | None"]


@dataclass
class _Components:
    # Fields stay ``Any``: the configured strategy default is ``None`` (a Plan-2
    # seam), which would not satisfy the kernel's non-optional ``Components``
    # protocol. ``step``/``run`` always pass a concrete strategy via ``replace``.
    rollout: Any
    reward: Any
    strategy: Any
    criterion: Any


class Experiment:
    """One experiment: a workspace, a way to run, a way to score, a way to judge.

    Direct use: ``Experiment(ws, rollout=fn, reward=fn, criterion=fn)``.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        rollout: Any,
        reward: RewardFn = result_key,
        strategy: Strategy | None = None,
        criterion: Criterion | None = None,
        slice_id: str = "custom",
        instances: Sequence[Mapping[str, Any]] = (),
        seed: Mapping[str, str] | None = None,
        auto_promote: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self._components = _Components(
            rollout=rollout,
            reward=reward,
            strategy=strategy,
            criterion=criterion or improve("reward"),
        )
        self.slice = Slice(slice_id, tuple(instances))
        self.auto_promote = auto_promote
        self._ensure_seed(seed or {"prompt.md": ""})

    def _ensure_seed(self, seed: Mapping[str, str]) -> None:
        try:
            store.current(self.workspace)
        except FileNotFoundError:
            initial = store.stage(
                self.workspace,
                base=None,
                edits=dict(seed),
                manifest=Manifest(producer="experiment", note="initial"),
            )
            store.promote(self.workspace, initial)

    def current(self) -> Version:
        return store.current(self.workspace)

    def step(self, strategy: Strategy) -> Decision | None:
        components = replace(self._components, strategy=strategy)
        return loop.step(
            self.workspace, components, self.slice, auto_promote=self.auto_promote
        )

    def run(self, strategy: Strategy, *, n: int = 1) -> list[Decision]:
        components = replace(self._components, strategy=strategy)
        return loop.run(
            self.workspace,
            components,
            self.slice,
            n=n,
            auto_promote=self.auto_promote,
        )

    def history(self, *, limit: int | None = None) -> str:
        rows = log.read(self.workspace, limit=limit)
        if not rows:
            return "no decisions yet"
        return "\n".join(f"{d.id} [{d.kind}] {d.outcome}: {d.reason}" for d in rows)

    def rollback(self) -> str:
        parent = store.current(self.workspace).parent
        if not parent:
            return "already at the initial version"
        store.promote(self.workspace, store.version(self.workspace, parent))
        return f"rolled back to {parent}"
