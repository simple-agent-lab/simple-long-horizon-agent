"""Experiment: the slim wirer. Holds the four components + workspace + slice and
exposes step / run / history / rollback. No policy lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key
from simple_agent_lab.evolution.kernel import log, loop, store
from simple_agent_lab.evolution.types import Decision, Manifest, Slice, Version


@dataclass
class _Components:
    rollout: Any
    reward: Any
    strategy: Any
    criterion: Any


class Experiment:
    """One experiment: a workspace, a way to run, a way to score, a way to judge.

    Level 1 (direct): ``Experiment(ws, rollout=fn, reward=fn, criterion=fn)``.
    Level 2 (config): ``Experiment.from_config(cfg)``.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        rollout: Any,
        reward: Any = result_key,
        criterion: Any | None = None,
        slice_id: str = "custom",
        instances: Sequence[Mapping[str, Any]] = (),
        seed: Mapping[str, str] | None = None,
        auto_promote: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self._components = _Components(
            rollout=rollout,
            reward=reward,
            strategy=None,
            criterion=criterion or improve("reward"),
        )
        self.slice = Slice(slice_id, tuple(instances))
        self.auto_promote = auto_promote
        self._ensure_seed(seed or {"prompt.md": ""})

    @classmethod
    def from_config(cls, config) -> "Experiment":
        from simple_agent_lab.evolution import registry

        exp = cls.__new__(cls)
        exp.workspace = Path(config.workspace)
        exp._components = _Components(
            rollout=registry.build("rollout", config.rollout),
            reward=registry.build("reward", config.reward),
            strategy=(
                registry.build("strategy", config.strategy)
                if config.strategy is not None
                else None
            ),
            criterion=registry.build("criterion", config.criterion),
        )
        exp.slice = Slice(config.slice_id, tuple(config.instances))
        exp.auto_promote = config.auto_promote
        exp._ensure_seed({"prompt.md": ""})
        return exp

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

    def step(self, strategy: Any) -> Decision | None:
        self._components.strategy = strategy
        return loop.step(
            self.workspace, self._components, self.slice, auto_promote=self.auto_promote
        )

    def run(self, strategy: Any, *, n: int = 1) -> list[Decision]:
        self._components.strategy = strategy
        return loop.run(
            self.workspace,
            self._components,
            self.slice,
            n=n,
            auto_promote=self.auto_promote,
        )

    def history(self, *, limit: int | None = None) -> str:
        rows = log.read(self.workspace, limit=limit)
        if not rows:
            return "no decisions yet"
        return "\n".join(
            f"{d.id} [{d.kind}] {d.outcome}: {d.reason}" for d in rows
        )

    def rollback(self) -> str:
        parent = store.current(self.workspace).parent
        if not parent:
            return "already at the initial version"
        store.promote(self.workspace, store.version(self.workspace, parent))
        return f"rolled back to {parent}"
