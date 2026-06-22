"""YAML config for the DGM SWE-bench recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PARENT_SELECTIONS = {"latest", "best", "score_prop", "score_child_prop"}


@dataclass(frozen=True)
class RunConfig:
    id: str
    output_root: str
    execute: bool = False
    reset: bool = False
    dotenv: str = ".env"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_path: str
    test_path: str


@dataclass(frozen=True)
class ExecutionConfig:
    parallel: int = 3
    max_turns: int = 75
    wheelhouse: str = "evals/out/swebench/wheelhouse/cp311-manylinux"
    uv_binary: str = ""


@dataclass(frozen=True)
class ModelConfig:
    api_kind: str = "openai-chat"
    model_env: str = "OPENAI_MODEL"
    default_model: str = "dgm-swebench"


@dataclass(frozen=True)
class DgmAlgorithmConfig:
    rounds: int = 4
    branches: int = 3
    meta_concurrency: int = 0
    parent_selection: str = "score_child_prop"


@dataclass(frozen=True)
class DgmConfig:
    run: RunConfig
    dataset: DatasetConfig
    execution: ExecutionConfig
    model: ModelConfig
    dgm: DgmAlgorithmConfig


def load_dgm_config(path: str | Path) -> DgmConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("DGM config must be a YAML mapping")
    _require(data, "run", "dataset", "execution", "model", "dgm")
    return DgmConfig(
        run=_run_config(_mapping(data["run"], "run")),
        dataset=_dataset_config(_mapping(data["dataset"], "dataset")),
        execution=_execution_config(_mapping(data["execution"], "execution")),
        model=_model_config(_mapping(data["model"], "model")),
        dgm=_dgm_config(_mapping(data["dgm"], "dgm")),
    )


def _require(mapping: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing required DGM config section: {key}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DGM config section {field!r} must be a mapping")
    return value


def _run_config(raw: Mapping[str, Any]) -> RunConfig:
    return RunConfig(
        id=str(raw["id"]),
        output_root=str(raw.get("output_root", "evals/out/dgm_swebench")),
        execute=bool(raw.get("execute", False)),
        reset=bool(raw.get("reset", False)),
        dotenv=str(raw.get("dotenv", ".env")),
    )


def _dataset_config(raw: Mapping[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        name=str(raw.get("name", "princeton-nlp/SWE-bench_Verified")),
        train_path=str(raw["train_path"]),
        test_path=str(raw["test_path"]),
    )


def _execution_config(raw: Mapping[str, Any]) -> ExecutionConfig:
    return ExecutionConfig(
        parallel=_positive_int(raw.get("parallel", 3), "execution.parallel"),
        max_turns=_positive_int(raw.get("max_turns", 75), "execution.max_turns"),
        wheelhouse=str(
            raw.get("wheelhouse", "evals/out/swebench/wheelhouse/cp311-manylinux")
        ),
        uv_binary=str(raw.get("uv_binary", "")),
    )


def _model_config(raw: Mapping[str, Any]) -> ModelConfig:
    api_kind = str(raw.get("api_kind", "openai-chat"))
    if api_kind not in {"openai-chat", "openai-responses"}:
        raise ValueError("model.api_kind must be openai-chat or openai-responses")
    return ModelConfig(
        api_kind=api_kind,
        model_env=str(raw.get("model_env", "OPENAI_MODEL")),
        default_model=str(raw.get("default_model", "dgm-swebench")),
    )


def _dgm_config(raw: Mapping[str, Any]) -> DgmAlgorithmConfig:
    parent_selection = str(raw.get("parent_selection", "score_child_prop"))
    if parent_selection not in PARENT_SELECTIONS:
        raise ValueError(
            "dgm.parent_selection must be latest, best, score_prop, or "
            f"score_child_prop; got {parent_selection!r}"
        )
    branches = _positive_int(raw.get("branches", 3), "dgm.branches")
    parallel_hint = raw.get("parallel")
    if parallel_hint is not None:
        raise ValueError("dgm.parallel is not supported; use execution.parallel")
    return DgmAlgorithmConfig(
        rounds=_positive_int(raw.get("rounds", 4), "dgm.rounds"),
        branches=branches,
        meta_concurrency=_nonnegative_int(
            raw.get("meta_concurrency", 0), "dgm.meta_concurrency"
        ),
        parent_selection=parent_selection,
    )


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer; got {value!r}") from None
    if parsed < 1:
        raise ValueError(f"{field} must be >= 1; got {value!r}")
    return parsed


def _nonnegative_int(value: object, field: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"{field} must be a non-negative integer; got {value!r}"
        ) from None
    if parsed < 0:
        raise ValueError(f"{field} must be >= 0; got {value!r}")
    return parsed
