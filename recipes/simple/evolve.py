"""Simple self-evolving SWE-bench recipe.

The framework is good enough that this short script starts a real self-evolving
training run: a model rewrites the whole agent program under ``agent/``, the
evolution kernel compares each candidate on a train slice in a SWE-bench Docker
sandbox, and the best valid agent is scored on a held-out test slice. Dry by
default; --execute runs real model + Docker.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for `recipes._shared`
sys.path.insert(0, str(ROOT / "src"))

from recipes import _shared  # noqa: E402
from evals.swebench import evolution_adapter as er  # noqa: E402
from simple_agent_lab.evolution import Experiment  # noqa: E402
from simple_agent_lab.evolution.components.criterion import promote_not_worse  # noqa: E402
from simple_agent_lab.evolution.components.strategy import (  # noqa: E402
    model_program_strategy,
)
from simple_agent_lab.evolution.types import Slice, Version  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402

SYSTEM_PROMPT = """You are a meta-agent evolving a SWE-bench coding agent.
The agent is a Python package under `agent/`; `agent/agent_program.py` defines
`build_agent(*, provider, cwd, base_system_prompt) -> Agent`. Edit any file under
`agent/` (full file contents, AST-valid). Keep build_agent present. Return ONLY
JSON: {"note":"...","evidence":["..."],"edits":{"agent/<path>":"FULL"|null}}.
Make one focused change likely to raise the resolve rate.
"""


def run(args: argparse.Namespace) -> None:
    _shared.load_dotenv(args.dotenv)
    model = os.environ.get("OPENAI_MODEL", "evolving-swebench")
    layout = er.PerformanceLayout(Path(args.output_root), args.run_id)
    train = er.load_dataset(args.train_dataset)
    test = er.load_dataset(args.test_dataset)
    print(
        f"run root: {layout.run_root}\n"
        f"train: {len(train)}  test: {len(test)}  rounds: {args.rounds}"
    )
    if not args.execute:
        print("\ndry run only; pass --execute to run model + Docker evolution")
        return

    layout.create()
    _shared.check_docker_available()
    base = er.build_swebench_rollout(
        layout,
        concurrency=1,
        run_kwargs={"api_kind": "openai-chat", "max_turns": args.max_turns},
        wheelhouse=args.wheelhouse,
        uv_binary=args.uv_binary,
        in_env_scoring=True,
        version_artifacts=er.version_package_artifacts,
        container_module=er.EVOLVING_CONTAINER_MODULE,
    )
    rollout = er.make_scaffold_rollout(
        base, dataset_name=er.DEFAULT_DATASET, model_name=model
    )
    provider = Provider(
        id="openai-chat",
        api="openai-chat",
        model=model,
        base_url=os.environ.get(er.OPENAI_BASE_URL_ENV) or None,
        api_key_env=er.OPENAI_AUTH_ENV,
    )
    exp = Experiment(
        layout.evolution_workspace,
        rollout=rollout,
        reward=er.swebench_reward,
        criterion=promote_not_worse("reward", tol=args.promotion_tolerance),
        slice_id="swebench-train",
        instances=train,
        seed=er.seed_files(
            model=model,
            api_kind="openai-chat",
            base_url=os.environ.get(er.OPENAI_BASE_URL_ENV, ""),
        ),
    )
    strategy = model_program_strategy(
        provider=provider,
        prefix="agent/",
        system_prompt=SYSTEM_PROMPT,
        parent_selection="current",
    )
    decisions = exp.run(strategy, n=args.rounds)
    print(f"\ncompleted {len(decisions)} generations; current={exp.current().hash}")
    run_heldout_scoring(args, layout, test, exp.current(), base, model_name=model)


def heldout_slice(
    version: Version, test_records: Sequence[Mapping[str, object]]
) -> Slice:
    del version  # agent package reaches the container via version_artifacts staging
    return Slice("swebench-test", tuple(dict(rec) for rec in test_records))


def heldout_run_id(
    version: Version, test_records: Sequence[Mapping[str, object]]
) -> str:
    slice_ = heldout_slice(version, test_records)
    return f"{version.hash}-{slice_.sha}"


def run_heldout_scoring(
    args: argparse.Namespace,
    layout: er.PerformanceLayout,
    test_records: Sequence[Mapping[str, object]],
    version: Version,
    base_rollout,
    *,
    model_name: str,
) -> None:
    print("\nheld-out test rollout:")
    test_slice = heldout_slice(version, test_records)
    base_rollout(version, test_slice)
    source_run_id = heldout_run_id(version, test_records)
    print(f"test run id: {source_run_id}")
    collect = er.collect_predictions_command(
        layout,
        dataset_name=er.DEFAULT_DATASET,
        model_name=model_name,
        source_run_id=source_run_id,
    )
    official = er.official_eval_command(
        layout,
        dataset_name=er.DEFAULT_DATASET,
        instance_ids=er.instance_ids(test_records),
        max_workers=1,
    )
    print("\ncollect predictions:")
    print(" ".join(collect))
    subprocess.run(collect, cwd=ROOT, check=True)
    print("\nofficial scoring:")
    print(" ".join(official))
    subprocess.run(official, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--test-dataset", required=True)
    p.add_argument("--output-root", default="evals/out/self_evolving/simple")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=75)
    p.add_argument("--promotion-tolerance", type=float, default=0.0)
    p.add_argument(
        "--wheelhouse", default="evals/out/swebench/wheelhouse/cp311-manylinux"
    )
    p.add_argument("--uv-binary", default="")
    p.add_argument("--dotenv", default=".env")
    p.add_argument("--execute", action="store_true")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
