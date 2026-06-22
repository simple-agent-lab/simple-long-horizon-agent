"""Generic CLI entry point for configured self-evolving runs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from simple_agent_lab.evals.instances import InstanceSet
from simple_agent_lab.evolution.config import (
    SelfEvolvingConfig,
    SelfEvolvingRun,
    build_self_evolving_run,
    load_self_evolving_config,
    safe_run_root,
)
from simple_agent_lab.evolution.experiment import Strategy
from simple_agent_lab.evolution.kernel.loop import means, score
from simple_agent_lab.evolution.types import Decision, Run, RunScores, Version

EvaluationRow = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect a generic self-evolving agent experiment."
    )
    parser.add_argument("--config", help="Path to the YAML run config.")
    parser.add_argument("--run-id", help="Override run.id before building.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the configured evolution loop instead of printing the dry-run plan.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear stale state for this run before building.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Print the run root that external monitors should watch.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config:
        parser.error("--config is required")
    config = _apply_overrides(load_self_evolving_config(args.config), args)
    _load_dotenv(config.run.dotenv)
    run_root = safe_run_root(config.run.output_root, config.run.id)

    if args.monitor:
        print(f"monitor: {run_root}")
        return 0

    built = build_self_evolving_run(config)

    if config.run.execute:
        decisions, report_path = _execute(config, built)
        print(
            "completed: "
            f"run_id={config.run.id} "
            f"rounds={config.evolution.rounds} "
            f"decisions={len(decisions)}"
        )
        if report_path is not None:
            print(f"evaluation report: {report_path}")
        return 0

    _print_dry_run_plan(config)
    return 0


def _apply_overrides(
    config: SelfEvolvingConfig, args: argparse.Namespace
) -> SelfEvolvingConfig:
    run = config.run
    if args.run_id:
        run = replace(run, id=args.run_id)
    if args.execute:
        run = replace(run, execute=True)
    if args.reset:
        run = replace(run, reset=True)
    return replace(config, run=run)


def _load_dotenv(path: str | Path) -> None:
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def _execute(
    config: SelfEvolvingConfig, built: SelfEvolvingRun
) -> tuple[list[Decision], Path | None]:
    _validate_evaluation_config(config, built.heldout)
    strategy = cast(Strategy, built.strategy)
    evaluations: list[EvaluationRow] = []

    if config.evaluation.baseline_heldout:
        row = _evaluate_heldout(built, "baseline", built.experiment.current())
        evaluations.append(row)
        _print_evaluation(row)

    decisions: list[Decision] = []
    for round_index in range(1, config.evolution.rounds + 1):
        decision = built.experiment.step(strategy)
        if decision is not None:
            decisions.append(decision)
        every = config.evaluation.heldout_every_rounds
        if every and round_index % every == 0:
            row = _evaluate_heldout(
                built,
                f"round-{round_index}",
                built.experiment.current(),
            )
            evaluations.append(row)
            _print_evaluation(row)

    if config.evaluation.final_heldout:
        row = _evaluate_heldout(built, "final", built.experiment.current())
        evaluations.append(row)
        _print_evaluation(row)

    if not evaluations:
        return decisions, None

    summary = _evaluation_summary(config, evaluations)
    _print_delta(summary["delta"])
    report_path = _write_evaluation_summary(config, summary)
    return decisions, report_path


def _validate_evaluation_config(
    config: SelfEvolvingConfig, heldout: InstanceSet | None
) -> None:
    evaluation = config.evaluation
    if evaluation.heldout_every_rounds < 0:
        raise ValueError("evaluation.heldout_every_rounds must be >= 0")
    if evaluation.repeats != 1:
        raise ValueError(
            "evaluation.repeats is not supported by the generic simple runner yet; "
            "use repeats: 1"
        )
    if evaluation.official_scoring:
        raise ValueError(
            "evaluation.official_scoring is not a generic simple-runner switch; "
            "configure scoring on the selected suite instead"
        )
    enabled = (
        evaluation.baseline_heldout
        or evaluation.final_heldout
        or evaluation.heldout_every_rounds > 0
    )
    if enabled and heldout is None:
        raise ValueError(
            "instances.heldout is required when heldout evaluation is enabled"
        )


def _evaluate_heldout(
    built: SelfEvolvingRun, label: str, version: Version
) -> EvaluationRow:
    if built.heldout is None:
        raise ValueError("instances.heldout is required for heldout evaluation")
    runs = tuple(built.rollout(version, built.heldout))
    run_scores = score(runs, built.reward)
    return {
        "label": label,
        "version": version.hash,
        "slice": {
            "id": built.heldout.id,
            "sha": built.heldout.sha,
            "count": len(built.heldout.instances),
        },
        "metrics": _metrics(runs, run_scores),
        "runs": [
            _run_summary(run, run_scores.get(run.instance_id, {})) for run in runs
        ],
    }


def _metrics(runs: Sequence[Run], run_scores: RunScores) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "total": len(runs),
        "ok": sum(1 for run in runs if run.ok),
    }
    for dim, value in means(run_scores).items():
        metrics[f"{dim}_mean"] = value

    resolved = [
        bool(run.result["resolved"]) for run in runs if "resolved" in run.result
    ]
    if resolved:
        count = sum(1 for value in resolved if value)
        metrics["resolved"] = count
        metrics["resolved_total"] = len(resolved)
        metrics["resolved_rate"] = count / len(resolved)
    return metrics


def _run_summary(run: Run, scores: Mapping[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instance_id": run.instance_id,
        "ref": run.ref,
        "ok": run.ok,
        "scores": dict(scores),
    }
    result = run.result
    if "resolved" in result:
        row["resolved"] = bool(result["resolved"])
    if "score" in result:
        row["score"] = result["score"]
    return row


def _evaluation_summary(
    config: SelfEvolvingConfig, evaluations: Sequence[EvaluationRow]
) -> dict[str, Any]:
    return {
        "schema": "simple-agent-lab.evolution.performance.v1",
        "run_id": config.run.id,
        "evaluations": list(evaluations),
        "delta": _evaluation_delta(evaluations),
    }


def _evaluation_delta(evaluations: Sequence[EvaluationRow]) -> dict[str, float | int]:
    baseline = next((row for row in evaluations if row["label"] == "baseline"), None)
    final = next((row for row in evaluations if row["label"] == "final"), None)
    if baseline is None or final is None:
        return {}

    skip = {"total", "ok", "resolved_total"}
    base_metrics = baseline["metrics"]
    final_metrics = final["metrics"]
    out: dict[str, float | int] = {}
    for key in sorted(set(base_metrics) & set(final_metrics) - skip):
        base_value = base_metrics[key]
        final_value = final_metrics[key]
        if isinstance(base_value, (int, float)) and isinstance(
            final_value, (int, float)
        ):
            delta = final_value - base_value
            out[key] = int(delta) if isinstance(final_value, int) else float(delta)
    return out


def _write_evaluation_summary(
    config: SelfEvolvingConfig, summary: dict[str, Any]
) -> Path:
    root = safe_run_root(config.run.output_root, config.run.id) / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _print_evaluation(row: EvaluationRow) -> None:
    metrics = row["metrics"]
    parts = [
        f"heldout {row['label']}:",
        f"reward={float(metrics.get('reward_mean', 0.0)):.3f}",
    ]
    if "resolved" in metrics:
        parts.append(
            f"resolved={int(metrics['resolved'])}/{int(metrics['resolved_total'])}"
        )
    print(" ".join(parts))


def _print_delta(delta: dict[str, float | int]) -> None:
    if not delta:
        return
    parts = []
    if "reward_mean" in delta:
        parts.append(f"reward={float(delta['reward_mean']):+.3f}")
    if "resolved" in delta:
        parts.append(f"resolved={int(delta['resolved']):+d}")
    if parts:
        print("heldout delta: " + " ".join(parts))


def _print_dry_run_plan(config: SelfEvolvingConfig) -> None:
    run_root = safe_run_root(config.run.output_root, config.run.id)
    print("dry-run self-evolving plan")
    print(f"run id: {config.run.id}")
    print(f"run root: {run_root}")
    print(f"suite: {config.suite.name}")
    print(f"surface: {config.surface.name}")
    print(f"editable components: {', '.join(config.surface.editable_components)}")
    print(f"train: {config.instances.train.id}")
    print(f"train count: {_count_jsonl_rows(config.instances.train.path)}")
    if config.instances.heldout:
        print(f"heldout: {config.instances.heldout.id}")
        print(f"heldout count: {_count_jsonl_rows(config.instances.heldout.path)}")
    print(f"rounds: {config.evolution.rounds}")


def _count_jsonl_rows(path: str) -> int:
    return sum(
        1
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
