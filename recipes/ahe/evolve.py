"""Run the config-backed AHE self-evolving SWE-bench recipe."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from simple_agent_lab.evolution.run import main as run_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "ahe_swebench.yaml"
EVOLVING_CONTAINER_MODULE = "simple_agent_lab.evals.suites.swebench.evolving"


def register_recipe_factories(config_path: str | Path | None = None) -> None:
    """Register the YAML-named factories used by the AHE recipe."""

    from evals.swebench.suite import SwebenchSuite
    from recipes.ahe.strategy import ahe_model_strategy
    from recipes.ahe.surface import ahe_harness_surface
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evolution import registry

    def swebench_suite(**args):
        suite = SwebenchSuite(**args)
        suite.container_module = EVOLVING_CONTAINER_MODULE
        return suite

    registry.SUITES.setdefault("swebench", swebench_suite)
    registry.SURFACES.setdefault(
        "ahe_harness_surface",
        lambda *, artifact_key, **_args: ahe_harness_surface(artifact_key=artifact_key),
    )
    registry.BACKENDS.setdefault(
        "local_docker", lambda **args: LocalDockerBackend(**args)
    )
    registry.STORES.setdefault("local_dir", lambda root, **_args: LocalDirStore(root))
    registry.STRATEGIES.setdefault("ahe_model", ahe_model_strategy)
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
    register_recipe_factories(config_path)
    if config_path is not None:
        _preflight_if_execute(args, config_path)
    return run_main(args)


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
