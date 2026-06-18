"""Run the config-backed simple self-evolving SWE-bench recipe."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.swebench.self_evolving import (  # noqa: E402
    register_swebench_self_evolving_factories,
)
from simple_agent_lab.evolution.run import main as run_main  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "simple_swebench.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    register_swebench_self_evolving_factories()
    if not _has_option(args, "--config") and not _asks_for_help(args):
        args = ["--config", str(DEFAULT_CONFIG), *args]
    return run_main(args)


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _asks_for_help(args: Sequence[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in args)


if __name__ == "__main__":
    raise SystemExit(main())
