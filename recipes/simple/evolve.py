"""Run the config-backed simple self-evolving SWE-bench recipe."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from simple_agent_lab.evolution.run import main as run_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "simple_swebench.yaml"
EVOLVING_CONTAINER_MODULE = "simple_agent_lab.evals.suites.swebench.evolving"


def register_recipe_factories(config_path: str | Path | None = None) -> None:
    """Register the factories this runnable recipe names in YAML.

    This keeps SWE-bench represented by its existing ``SwebenchSuite`` while the
    recipe wires the concrete backend, store, surface, and strategy choices.
    """

    from evals.swebench.suite import SwebenchSuite
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evolution import registry
    from simple_agent_lab.evolution.components.repo_strategy import (
        source_tree_agent_strategy,
    )
    from simple_agent_lab.evolution.source_tree import source_tree_agent_surface

    def swebench_suite(**args):
        suite = SwebenchSuite(**args)
        suite.container_module = EVOLVING_CONTAINER_MODULE
        return suite

    registry.SUITES.setdefault("swebench", swebench_suite)
    registry.SURFACES.setdefault(
        "source_tree",
        lambda *, artifact_key, **_args: source_tree_agent_surface(
            repo_root=ROOT,
            artifact_key=artifact_key,
        ),
    )
    registry.BACKENDS.setdefault(
        "local_docker", lambda **args: LocalDockerBackend(**args)
    )
    registry.STORES.setdefault("local_dir", lambda root, **_args: LocalDirStore(root))
    registry.STRATEGIES.setdefault("source_tree_agent", source_tree_agent_strategy)
    if config_path is not None:
        reward = _swebench_reward_from_config(config_path)
        registry.REWARDS["result_key"] = lambda reward=reward: reward


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--config") and _asks_for_help(args):
        print(f"default config: {DEFAULT_CONFIG.relative_to(ROOT)}")
    if not _has_option(args, "--config") and not _asks_for_help(args):
        args = ["--config", str(DEFAULT_CONFIG), *args]
    config_path = None if _asks_for_help(args) else _option_value(args, "--config")
    normalized_config = None
    if config_path is not None:
        args, config_path, normalized_config = _normalize_config_args(args, config_path)
    try:
        register_recipe_factories(config_path)
        if config_path is not None:
            _preflight_if_execute(args, config_path)
        return run_main(args)
    finally:
        if normalized_config is not None:
            normalized_config.cleanup()


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _asks_for_help(args: Sequence[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in args)


def _option_value(args: Sequence[str], option: str) -> str | None:
    for i, arg in enumerate(args):
        if arg == option and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
    return None


def _normalize_config_args(
    args: Sequence[str], config_path: str
) -> tuple[list[str], str, tempfile.TemporaryDirectory[str]]:
    path = Path(config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    base = ROOT if path == DEFAULT_CONFIG.resolve(strict=False) else path.parent
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    config_data = cast("dict[str, Any]", data)

    _normalize_instance_paths(config_data, base=base)
    _normalize_source_tree_repo_root(config_data)
    _normalize_repo_relative_backend_paths(config_data)

    temp = tempfile.TemporaryDirectory(prefix="sal-simple-config-")
    normalized = Path(temp.name) / path.name
    normalized.write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )
    return (
        _replace_option_value(args, "--config", str(normalized)),
        str(normalized),
        temp,
    )


def _replace_option_value(args: Sequence[str], option: str, value: str) -> list[str]:
    out = list(args)
    for i, arg in enumerate(out):
        if arg == option and i + 1 < len(out):
            out[i + 1] = value
            return out
        if arg.startswith(f"{option}="):
            out[i] = f"{option}={value}"
            return out
    return [option, value, *out]


def _normalize_instance_paths(data: dict[str, Any], *, base: Path) -> None:
    instances = data.get("instances")
    if not isinstance(instances, dict):
        return
    instances = cast("dict[str, Any]", instances)
    for name in ("train", "heldout"):
        item = instances.get(name)
        if not isinstance(item, dict):
            continue
        item = cast("dict[str, Any]", item)
        raw_path = item.get("path")
        if isinstance(raw_path, str):
            item["path"] = _absolute_path(raw_path, base=base)


def _normalize_source_tree_repo_root(data: dict[str, Any]) -> None:
    surface = data.get("surface")
    if not isinstance(surface, dict):
        return
    surface = cast("dict[str, Any]", surface)
    if surface.get("name") != "source_tree":
        return
    strategy = data.setdefault("strategy", {})
    if not isinstance(strategy, dict):
        return
    strategy = cast("dict[str, Any]", strategy)
    args = strategy.setdefault("args", {})
    if not isinstance(args, dict):
        return
    args = cast("dict[str, Any]", args)
    raw = args.get("repo_root", ".")
    if isinstance(raw, str):
        args["repo_root"] = _absolute_path(raw, base=ROOT)


def _normalize_repo_relative_backend_paths(data: dict[str, Any]) -> None:
    execution = data.get("execution")
    if not isinstance(execution, dict):
        return
    execution = cast("dict[str, Any]", execution)
    backend = execution.get("backend")
    if not isinstance(backend, dict):
        return
    backend = cast("dict[str, Any]", backend)
    args = backend.get("args")
    if not isinstance(args, dict):
        return
    args = cast("dict[str, Any]", args)
    for key in ("wheelhouse", "uv_binary"):
        raw = args.get(key)
        if isinstance(raw, str):
            args[key] = _absolute_path(raw, base=ROOT)


def _absolute_path(path: str, *, base: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve(strict=False))


def _swebench_reward_from_config(config_path: str | Path):
    from evals.swebench import harness
    from recipes import swebench_reward
    from simple_agent_lab.evolution.config import load_self_evolving_config

    config = load_self_evolving_config(config_path)
    instance_paths = [config.instances.train.path]
    if config.instances.heldout is not None:
        instance_paths.append(config.instances.heldout.path)
    dataset_name = str(config.suite.args.get("dataset_name") or harness.DEFAULT_DATASET)
    model_name = os.environ.get(config.model.model_env, "") or "simple-agent-lab"
    return swebench_reward.make_reuse_reward_from_paths(
        instance_paths=instance_paths,
        dataset_name=dataset_name,
        model_name=model_name,
    )


def _preflight_if_execute(args: Sequence[str], config_path: str | Path) -> None:
    from evals.swebench.suite import SwebenchSuite
    from recipes import runtime as recipe_runtime
    from simple_agent_lab.evals.instances import load_jsonl_instances
    from simple_agent_lab.evolution.config import load_self_evolving_config

    config = load_self_evolving_config(config_path)
    execute = config.run.execute or _has_option(args, "--execute")
    if not execute or _has_option(args, "--monitor") or config.suite.name != "swebench":
        return
    suite = SwebenchSuite(**config.suite.args)
    pull = str(config.execution.backend.args.get("pull", "missing"))
    recipe_runtime.preflight_suite_images(
        suite,
        load_jsonl_instances(config.instances.train.path),
        pull=pull,
        label="train",
    )
    if config.instances.heldout is not None and (
        config.evaluation.baseline_heldout
        or config.evaluation.final_heldout
        or config.evaluation.heldout_every_rounds
    ):
        recipe_runtime.preflight_suite_images(
            suite,
            load_jsonl_instances(config.instances.heldout.path),
            pull=pull,
            label="heldout",
        )


if __name__ == "__main__":
    raise SystemExit(main())
