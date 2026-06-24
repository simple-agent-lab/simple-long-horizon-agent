"""Run the config-backed AHE self-evolving SWE-bench recipe."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

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
    from simple_agent_lab.evolution import run as generic_run
    from simple_agent_lab.evolution.config import (
        build_self_evolving_run,
        load_self_evolving_config,
    )

    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--config") and _asks_for_help(args):
        print(f"default config: {DEFAULT_CONFIG.relative_to(ROOT)}")
    if not _has_option(args, "--config") and not _asks_for_help(args):
        args = ["--config", str(DEFAULT_CONFIG), *args]
    config_path = None if _asks_for_help(args) else _option_value(args, "--config")
    register_recipe_factories(config_path)
    if config_path is not None:
        _preflight_if_execute(args, config_path)
    if _asks_for_help(args):
        return run_main(args)

    parsed = generic_run.build_parser().parse_args(args)
    config = generic_run._apply_overrides(
        load_self_evolving_config(parsed.config), parsed
    )
    generic_run._load_dotenv(config.run.dotenv)
    if parsed.monitor or not config.run.execute:
        return run_main(args)

    built = build_self_evolving_run(config)
    decisions, report_path = _run_ahe_execute(config, built)
    print(
        "completed: "
        f"run_id={config.run.id} "
        f"rounds={config.evolution.rounds} "
        f"decisions={len(decisions)}"
    )
    if report_path is not None:
        print(f"evaluation report: {report_path}")
    return 0


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


def _run_ahe_execute(config, built) -> tuple[list[Any], Path | None]:
    from recipes.ahe import ledger
    from simple_agent_lab.evolution import run as generic_run
    from simple_agent_lab.evolution.experiment import Strategy

    generic_run._validate_evaluation_config(config, built.heldout)
    strategy = cast(Strategy, built.strategy)
    evaluations: list[dict[str, Any]] = []
    run_root = Path(built.experiment.workspace).parent

    if config.evaluation.baseline_heldout:
        row = generic_run._evaluate_heldout(
            built, "baseline", built.experiment.current()
        )
        evaluations.append(row)
        generic_run._print_evaluation(row)

    decisions = []
    for round_index in range(1, config.evolution.rounds + 1):
        decision = built.experiment.step(strategy)
        if decision is None:
            ledger.append_history(run_root, f"- round {round_index}: no proposal")
            continue
        decisions.append(decision)
        _write_round_ledger(run_root, round_index, decision, built.reward)
        every = config.evaluation.heldout_every_rounds
        if every and round_index % every == 0:
            row = generic_run._evaluate_heldout(
                built,
                f"round-{round_index}",
                built.experiment.current(),
            )
            evaluations.append(row)
            generic_run._print_evaluation(row)

    if config.evaluation.final_heldout:
        row = generic_run._evaluate_heldout(
            built, "final", built.experiment.current()
        )
        evaluations.append(row)
        generic_run._print_evaluation(row)

    if not evaluations:
        return decisions, None

    summary = generic_run._evaluation_summary(config, evaluations)
    generic_run._print_delta(summary["delta"])
    report_path = generic_run._write_evaluation_summary(config, summary)
    return decisions, report_path


def _write_round_ledger(run_root: Path, round_index: int, decision, reward) -> None:
    from recipes.ahe import ledger

    round_path = ledger.round_dir(run_root, round_index)
    manifest = _manifest_for_round(round_path, round_index)
    baseline_scores = _scores_for_run_id(
        run_root, str(decision.runs.get("baseline", "")), reward
    )
    candidate_scores = _scores_for_run_id(
        run_root, str(decision.runs.get("candidate", "")), reward
    )
    change_evaluation = ledger.evaluate_manifest_predictions(
        manifest,
        baseline_scores,
        candidate_scores,
    )
    ledger.write_json(round_path / "change_evaluation.json", change_evaluation)
    ledger.update_task_history(run_root, round_index, candidate_scores)
    ledger.update_best_ever(
        run_root,
        round_index,
        str(decision.candidate.get("hash", "")),
        candidate_scores,
    )
    outcome = "accepted" if decision.accepted else "rejected"
    ledger.append_history(
        run_root,
        (
            f"- round {round_index}: "
            f"{decision.candidate.get('hash', '')} {outcome} ({decision.reason})"
        ),
    )


def _manifest_for_round(round_path: Path, round_index: int) -> Mapping[str, object]:
    from recipes.ahe import ledger

    manifest = ledger.read_json(
        round_path / "change_manifest.json",
        default={"round": round_index, "changes": []},
    )
    if isinstance(manifest, Mapping):
        return manifest
    return {"round": round_index, "changes": []}


def _scores_for_run_id(
    run_root: Path, run_id: str, reward
) -> dict[str, dict[str, float]]:
    from simple_agent_lab.evolution.kernel.loop import score
    from simple_agent_lab.evolution.types import Run

    if not run_id:
        return {}
    path = run_root / "runs" / run_id
    if not path.is_dir():
        return {}
    runs = [Run(entry) for entry in sorted(path.iterdir()) if entry.is_dir()]
    return score(runs, reward)


if __name__ == "__main__":
    raise SystemExit(main())
