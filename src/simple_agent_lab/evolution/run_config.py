"""General self-evolving YAML config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    id: str
    output_root: str
    execute: bool = False
    reset: bool = False
    dotenv: str = ".env"


@dataclass(frozen=True)
class NamedConfig:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceConfig:
    name: str
    editable_components: tuple[str, ...]
    default: str
    artifact_key: str


@dataclass(frozen=True)
class InstanceFileConfig:
    id: str
    path: str


@dataclass(frozen=True)
class InstancesConfig:
    train: InstanceFileConfig
    heldout: InstanceFileConfig | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    backend: NamedConfig
    store: NamedConfig
    parallel: int | str = 1
    max_turns: int = 75


@dataclass(frozen=True)
class ModelConfig:
    api_kind: str
    model_env: str
    api_key_env: str
    base_url_env: str = ""


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CriterionConfig:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvolutionRunConfig:
    algorithm: str
    rounds: int
    criterion: CriterionConfig
    branches: int = 1
    meta_concurrency: int = 1
    parent_selection: str = "current"


@dataclass(frozen=True)
class EvaluationConfig:
    baseline_heldout: bool = False
    final_heldout: bool = False
    heldout_every_rounds: int = 0
    repeats: int = 1
    official_scoring: bool = False


@dataclass(frozen=True)
class SelfEvolvingConfig:
    run: RunConfig
    suite: NamedConfig
    surface: SurfaceConfig
    instances: InstancesConfig
    execution: ExecutionConfig
    model: ModelConfig
    strategy: StrategyConfig
    evolution: EvolutionRunConfig
    evaluation: EvaluationConfig


def load_self_evolving_config(path: str | Path) -> SelfEvolvingConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    _require(
        data,
        "run",
        "suite",
        "surface",
        "instances",
        "execution",
        "model",
        "strategy",
        "evolution",
        "evaluation",
    )
    return SelfEvolvingConfig(
        run=_run_config(data["run"]),
        suite=_named_config(data["suite"]),
        surface=_surface_config(data["surface"]),
        instances=_instances_config(data["instances"]),
        execution=_execution_config(data["execution"]),
        model=ModelConfig(**data["model"]),
        strategy=StrategyConfig(
            name=str(data["strategy"]["name"]),
            args=dict(data["strategy"].get("args", {})),
        ),
        evolution=_evolution_config(data["evolution"]),
        evaluation=EvaluationConfig(**data["evaluation"]),
    )


def _require(mapping: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing required config section: {key}")


def _named_config(raw: Mapping[str, Any]) -> NamedConfig:
    return NamedConfig(name=str(raw["name"]), args=dict(raw.get("args", {})))


def _run_config(raw: Mapping[str, Any]) -> RunConfig:
    return RunConfig(**raw)


def _surface_config(raw: Mapping[str, Any]) -> SurfaceConfig:
    return SurfaceConfig(
        name=str(raw["name"]),
        editable_components=tuple(raw.get("editable_components", ("everything",))),
        default=str(raw["default"]),
        artifact_key=str(raw["artifact_key"]),
    )


def _instances_config(raw: Mapping[str, Any]) -> InstancesConfig:
    heldout = raw.get("heldout")
    return InstancesConfig(
        train=InstanceFileConfig(**raw["train"]),
        heldout=InstanceFileConfig(**heldout) if heldout else None,
    )


def _execution_config(raw: Mapping[str, Any]) -> ExecutionConfig:
    return ExecutionConfig(
        backend=_named_config(raw["backend"]),
        store=_named_config(raw["store"]),
        parallel=raw.get("parallel", 1),
        max_turns=int(raw.get("max_turns", 75)),
    )


def _evolution_config(raw: Mapping[str, Any]) -> EvolutionRunConfig:
    criterion = raw.get(
        "criterion",
        {"name": "promote_not_worse", "args": {"dim": "reward"}},
    )
    return EvolutionRunConfig(
        algorithm=str(raw["algorithm"]),
        rounds=int(raw["rounds"]),
        criterion=CriterionConfig(
            name=str(criterion["name"]),
            args=dict(criterion.get("args", {})),
        ),
        branches=int(raw.get("branches", 1)),
        meta_concurrency=int(raw.get("meta_concurrency", 1)),
        parent_selection=str(raw.get("parent_selection", "current")),
    )
