"""Run the config-backed simple self-evolving SWE-bench recipe."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import recipes.runtime as recipe_runtime  # noqa: E402
from simple_agent_lab.evolution.config import load_self_evolving_config  # noqa: E402
from simple_agent_lab.evolution.run import main as run_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "simple_swebench.yaml"
EVOLVING_CONTAINER_MODULE = "simple_agent_lab.evals.suites.swebench.evolving"


def register_recipe_factories() -> None:
    """Register the factories this runnable recipe names in YAML.

    This keeps SWE-bench represented by its existing ``SwebenchSuite`` while the
    recipe wires the concrete backend, store, surface, and strategy choices.
    """

    from evals.swebench.suite import SwebenchSuite
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evals.suites.swebench import agent_package
    from simple_agent_lab.evolution import registry
    from simple_agent_lab.evolution.components.strategy import model_program_strategy
    from simple_agent_lab.evolution.surface import python_agent_surface

    def swebench_suite(**args):
        suite = SwebenchSuite(**args)
        suite.container_module = EVOLVING_CONTAINER_MODULE
        return suite

    registry.SUITES.setdefault("swebench", swebench_suite)
    registry.SURFACES.setdefault(
        "python_agent_package",
        lambda *, artifact_key, version_root="agent/", **_args: python_agent_surface(
            default_files=agent_package.default_agent_package(),
            artifact_key=artifact_key,
            version_root=version_root,
        ),
    )
    registry.BACKENDS.setdefault(
        "local_docker", lambda **args: LocalDockerBackend(**args)
    )
    registry.STORES.setdefault("local_dir", lambda root, **_args: LocalDirStore(root))
    registry.STRATEGIES.setdefault("model_program", model_program_strategy)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--config") and _asks_for_help(args):
        print(f"default config: {DEFAULT_CONFIG.relative_to(ROOT)}")
    if not _has_option(args, "--config") and not _asks_for_help(args):
        args = ["--config", str(DEFAULT_CONFIG), *args]
    if not _asks_for_help(args):
        _reject_example_datasets_if_execute(args)
    register_recipe_factories()
    return run_main(args)


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _asks_for_help(args: Sequence[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in args)


def _reject_example_datasets_if_execute(args: Sequence[str]) -> None:
    config_path = _option_value(args, "--config")
    if config_path is None:
        return
    config = load_self_evolving_config(config_path)
    if not (_has_option(args, "--execute") or config.run.execute):
        return
    paths: list[str | Path] = [config.instances.train.path]
    if config.instances.heldout is not None:
        paths.append(config.instances.heldout.path)
    recipe_runtime.reject_example_datasets_for_execute(
        paths,
        config_path=config_path,
    )


def _option_value(args: Sequence[str], option: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            return args[index + 1]
        prefix = f"{option}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
