"""General self-evolving YAML config."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, get_args

import yaml

from simple_agent_lab.evals.instances import InstanceSet, load_jsonl_instances
from simple_agent_lab.evolution import registry
from simple_agent_lab.evolution.components.rollout import Rollout, rollout_from_suite
from simple_agent_lab.evolution.experiment import Experiment
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.llm.provider import ApiKind, Provider


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


@dataclass(frozen=True)
class SelfEvolvingRun:
    config: SelfEvolvingConfig
    suite: object
    surface: AgentSurface
    editable_components: tuple[str, ...]
    train: InstanceSet
    heldout: InstanceSet | None
    rollout: Rollout
    strategy: object
    experiment: Experiment


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


def build_self_evolving_run(config: SelfEvolvingConfig) -> SelfEvolvingRun:
    run_root = Path(config.run.output_root) / config.run.id
    _validate_algorithm(config)
    provider = _provider(config)
    if config.run.reset:
        shutil.rmtree(run_root, ignore_errors=True)
    suite = registry.build("suite", _use(config.suite.name, **config.suite.args))
    surface = registry.build(
        "surface",
        _use(
            config.surface.name,
            default=config.surface.default,
            artifact_key=config.surface.artifact_key,
        ),
    )
    train = InstanceSet(
        config.instances.train.id,
        load_jsonl_instances(config.instances.train.path),
    )
    heldout = (
        InstanceSet(
            config.instances.heldout.id,
            load_jsonl_instances(config.instances.heldout.path),
        )
        if config.instances.heldout
        else None
    )
    backend = registry.build(
        "backend",
        _use(config.execution.backend.name, **config.execution.backend.args),
    )
    store = registry.build(
        "store",
        _use(
            config.execution.store.name,
            root=run_root / "runs",
            **config.execution.store.args,
        ),
    )
    rollout = rollout_from_suite(
        suite=suite,
        surface=surface,
        backend=backend,
        store=store,
        runs_root=run_root / "runs",
        concurrency=1
        if config.execution.parallel == "auto"
        else int(config.execution.parallel),
        run_kwargs={"max_turns": config.execution.max_turns},
    )
    strategy_args: dict[str, object] = {
        "provider": provider,
        "surface": surface,
        "editable_components": config.surface.editable_components,
    }
    strategy_args.update(config.strategy.args)
    strategy = registry.build(
        "strategy",
        _use(config.strategy.name, **strategy_args),
    )
    seed = {
        **surface.seed_files(),
        "provider.json": _provider_json(config),
    }
    experiment = Experiment(
        run_root / "evolution",
        rollout=rollout,
        reward=registry.build("reward", _use("result_key")),
        criterion=registry.build(
            "criterion",
            _use(config.evolution.criterion.name, **config.evolution.criterion.args),
        ),
        slice_id=train.id,
        instances=train.instances,
        seed=seed,
    )
    return SelfEvolvingRun(
        config=config,
        suite=suite,
        surface=surface,
        editable_components=config.surface.editable_components,
        train=train,
        heldout=heldout,
        rollout=rollout,
        strategy=strategy,
        experiment=experiment,
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
    editable_components = raw.get("editable_components", ("everything",))
    if isinstance(editable_components, str):
        components = (editable_components,)
    else:
        components = tuple(str(component) for component in editable_components)
    return SurfaceConfig(
        name=str(raw["name"]),
        editable_components=components,
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
    allowed = {"algorithm", "rounds", "criterion"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"unknown evolution config keys for simple v1 builder: {', '.join(unknown)}"
        )
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
    )


def _use(name: str, **args: object):
    from simple_agent_lab.evolution.config import Use

    return Use(name, **args)


def _validate_algorithm(config: SelfEvolvingConfig) -> None:
    if config.evolution.algorithm != "simple":
        raise ValueError(
            "unsupported evolution algorithm "
            f"{config.evolution.algorithm!r}; this builder supports only 'simple'"
        )


def _provider(config: SelfEvolvingConfig) -> Provider:
    api_kinds = set(get_args(ApiKind))
    if config.model.api_kind not in api_kinds:
        raise ValueError(f"unknown model api_kind {config.model.api_kind!r}")
    base_url = (
        os.environ.get(config.model.base_url_env, "").strip()
        if config.model.base_url_env
        else ""
    )
    return Provider(
        id=config.model.api_kind,
        api=cast(ApiKind, config.model.api_kind),
        model=os.environ.get(config.model.model_env, ""),
        base_url=base_url or None,
        api_key_env=config.model.api_key_env,
    )


def _provider_json(config: SelfEvolvingConfig) -> str:
    import json

    data = {
        "api": config.model.api_kind,
        "model": os.environ.get(config.model.model_env, ""),
        "api_key_env": config.model.api_key_env,
    }
    if config.model.base_url_env and os.environ.get(config.model.base_url_env):
        data["base_url"] = os.environ[config.model.base_url_env]
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
