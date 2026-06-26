"""Configured self-evolving runs.

Direct Python callers can instantiate ``Experiment(...)`` themselves. This
module owns the YAML-backed path: load the run config, resolve named factories
through ``registry``, and build the configured ``Experiment``.
"""

from __future__ import annotations

import json
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
from simple_agent_lab.evolution.registry import Use
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    candidate_source_artifacts,
)
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import RewardFn, Version
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
    args: Mapping[str, Any] = field(default_factory=dict)


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
    parallel: int = 1
    max_turns: int = 75
    run_kwargs: Mapping[str, Any] = field(default_factory=dict)


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
    reward: RewardFn
    strategy: object
    experiment: Experiment


def safe_run_root(output_root: str | Path, run_id: str) -> Path:
    """Return the resolved run root for one run id under an output root."""

    if not run_id:
        raise ValueError("unsafe run id: must be non-empty")
    if run_id in {".", ".."}:
        raise ValueError(f"unsafe run id {run_id!r}: must not be '.' or '..'")
    if "/" in run_id or "\\" in run_id:
        raise ValueError(f"unsafe run id {run_id!r}: must not contain path separators")

    run_path = Path(run_id)
    if run_path.is_absolute():
        raise ValueError(f"unsafe run id {run_id!r}: must be relative")

    root = Path(output_root).resolve(strict=False)
    candidate = (root / run_id).resolve(strict=False)
    if candidate.parent != root:
        raise ValueError(
            f"unsafe run id {run_id!r}: run root must be an immediate child of {root}"
        )
    return candidate


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
    run_root = safe_run_root(config.run.output_root, config.run.id)
    _validate_algorithm(config)
    provider = _provider(config)
    if config.run.reset:
        shutil.rmtree(run_root, ignore_errors=True)
    suite = registry.build("suite", Use(config.suite.name, **config.suite.args))
    surface = registry.build(
        "surface",
        Use(
            config.surface.name,
            default=config.surface.default,
            artifact_key=config.surface.artifact_key,
            **config.surface.args,
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
        Use(config.execution.backend.name, **config.execution.backend.args),
    )
    store = registry.build(
        "store",
        Use(
            config.execution.store.name,
            root=run_root / "runs",
            **config.execution.store.args,
        ),
    )
    source_tree_repo_root = (
        _source_tree_repo_root(config) if surface.id == "source_tree" else None
    )
    version_artifacts = None
    candidate_pythonpath: tuple[str, ...] = ()
    if source_tree_repo_root is not None:
        repo_root = source_tree_repo_root

        def source_tree_version_artifacts(version: Version) -> Mapping[str, bytes]:
            files = surface.files_from_version(version)
            return candidate_source_artifacts(repo_root, files)

        version_artifacts = source_tree_version_artifacts
        candidate_pythonpath = (CANDIDATE_SOURCE_CONTAINER_SRC,)
    rollout = rollout_from_suite(
        suite=suite,
        surface=surface,
        backend=backend,
        store=store,
        runs_root=run_root / "runs",
        concurrency=config.execution.parallel,
        run_kwargs={
            "max_turns": config.execution.max_turns,
            **dict(config.execution.run_kwargs),
        },
        version_artifacts=version_artifacts,
        candidate_pythonpath=candidate_pythonpath,
    )
    strategy_args: dict[str, object] = {
        "provider": provider,
        "surface": surface,
        "editable_components": config.surface.editable_components,
    }
    strategy_args.update(config.strategy.args)
    if surface.id == "source_tree" and "repo_root" not in strategy_args:
        strategy_args["repo_root"] = source_tree_repo_root or Path.cwd()
    strategy = registry.build(
        "strategy",
        Use(config.strategy.name, **strategy_args),
    )
    seed = {
        **surface.seed_files(),
        "provider.json": _provider_json(config),
    }
    reward = registry.build("reward", Use("result_key"))
    experiment = Experiment(
        run_root / "evolution",
        rollout=rollout,
        reward=reward,
        criterion=registry.build(
            "criterion",
            Use(config.evolution.criterion.name, **config.evolution.criterion.args),
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
        reward=reward,
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
        args=dict(raw.get("args", {})),
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
        parallel=_positive_int(raw.get("parallel", 1), "execution.parallel"),
        max_turns=int(raw.get("max_turns", 75)),
        run_kwargs=dict(raw.get("run_kwargs", {})),
    )


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer; got {value!r}") from None
    if parsed < 1:
        raise ValueError(f"{field} must be >= 1; got {value!r}")
    return parsed


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


def _validate_algorithm(config: SelfEvolvingConfig) -> None:
    if config.evolution.algorithm != "simple":
        raise ValueError(
            "unsupported evolution algorithm "
            f"{config.evolution.algorithm!r}; this builder supports only 'simple'"
        )


def _source_tree_repo_root(config: SelfEvolvingConfig) -> Path:
    raw = config.strategy.args.get("repo_root")
    return Path(str(raw)).resolve(strict=False) if raw is not None else Path.cwd()


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
    data = {
        "api": config.model.api_kind,
        "model": os.environ.get(config.model.model_env, ""),
        "api_key_env": config.model.api_key_env,
    }
    if config.model.base_url_env and os.environ.get(config.model.base_url_env):
        data["base_url"] = os.environ[config.model.base_url_env]
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
