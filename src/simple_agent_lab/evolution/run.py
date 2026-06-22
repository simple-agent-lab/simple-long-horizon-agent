"""Generic CLI entry point for configured self-evolving runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from simple_agent_lab.evolution.config import (
    SelfEvolvingConfig,
    build_self_evolving_run,
    load_self_evolving_config,
    safe_run_root,
)
from simple_agent_lab.evolution.experiment import Strategy


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
    run_root = safe_run_root(config.run.output_root, config.run.id)

    if args.monitor:
        print(f"monitor: {run_root}")
        return 0

    built = build_self_evolving_run(config)

    if config.run.execute:
        decisions = built.experiment.run(
            cast(Strategy, built.strategy),
            n=config.evolution.rounds,
        )
        print(
            "completed: "
            f"run_id={config.run.id} "
            f"rounds={config.evolution.rounds} "
            f"decisions={len(decisions)}"
        )
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
    print(f"rounds: {config.evolution.rounds}")


def _count_jsonl_rows(path: str) -> int:
    return sum(
        1
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
