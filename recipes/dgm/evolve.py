"""Faithful DGM self-evolving SWE-bench recipe.

A model-driven meta-agent rewrites the whole agent program under ``agent/``; the
evolution kernel runs a parallel open-ended archive-admission loop (branches per
round, best-valid promotion, archive parent selection), and the best-on-train
agent is scored on a held-out test split. All DGM knobs are exposed. Dry by
default; --execute runs real model + Docker.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for `recipes._shared`
sys.path.insert(0, str(ROOT / "src"))

from recipes import _shared  # noqa: E402
from simple_agent_lab.evals.suites.swebench import evolving_rollout as er  # noqa: E402
from simple_agent_lab.evolution import Experiment, archive, open_ended  # noqa: E402
from simple_agent_lab.evolution.components.criterion import valid_when  # noqa: E402
from simple_agent_lab.evolution.components.strategy import (  # noqa: E402
    model_program_strategy,
)
from simple_agent_lab.evolution.kernel import store as evo_store  # noqa: E402
from simple_agent_lab.evolution.types import Slice, Version  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("evals/out/dgm_swebench")
DEFAULT_MODEL = er.DEFAULT_MODEL_NAME

SYSTEM_PROMPT = """You are a meta-agent evolving a SWE-bench coding agent.
The agent is a Python package under `agent/`; `agent/agent_program.py` defines
`build_agent(*, provider, cwd, base_system_prompt) -> Agent`. Edit any file under
`agent/` (full file contents, AST-valid). Keep build_agent present. Return ONLY
JSON: {"note":"...","evidence":["..."],"edits":{"agent/<path>":"FULL"|null}}.
Make one focused change likely to raise the resolve rate.
"""


def run_workflow(args: argparse.Namespace) -> None:
    _shared.load_dotenv(args.dotenv)
    if args.model_name == DEFAULT_MODEL and os.environ.get("OPENAI_MODEL"):
        args.model_name = os.environ["OPENAI_MODEL"]
    output_root = Path(args.output_root)
    run_root = output_root / args.run_id
    workspace = run_root / "evolution"
    if args.reset and run_root.exists():
        _shared.cleanup_reset_containers(run_root)
        shutil.rmtree(run_root)
    layout = er.PerformanceLayout(output_root, args.run_id)
    layout.create()

    train_records = er.load_dataset(args.train_dataset)
    test_records = er.load_dataset(args.test_dataset)
    print_plan(args, layout, train_records, test_records)
    if not args.execute:
        print("\ndry run only; pass --execute to run model + Docker evolution")
        return

    _shared.check_docker_available()
    rounds, branches, meta_workers = resolve_schedule(args)
    resolution = _shared.resolve_parallel_workers(args.parallel, len(train_records))
    global_workers = resolution.workers
    per_branch = _shared.branch_concurrency(
        global_workers=global_workers, branches=branches
    )
    print(f"global workers: {global_workers} ({resolution.detail})")
    print(
        f"schedule: {rounds} rounds x {branches} branches "
        f"= {rounds * branches} candidates; meta-concurrency {meta_workers}"
    )
    print(
        f"per-branch rollout concurrency: {per_branch} "
        f"(<= {global_workers} containers total; memory is the hard cap)"
    )
    args.parallel = global_workers
    base_rollout = er.build_swebench_rollout(
        layout,
        dataset_name=args.dataset_name,
        concurrency=per_branch,
        run_kwargs={
            "api_kind": args.api_kind,
            "max_turns": args.max_turns,
        },
        wheelhouse=args.wheelhouse,
        uv_binary=args.uv_binary,
        in_env_scoring=True,
        version_artifacts=er.version_package_artifacts,
        container_module=er.EVOLVING_CONTAINER_MODULE,
    )
    graded_rollout = er.make_scaffold_rollout(
        base_rollout,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    provider = Provider(
        id="openai-chat",
        api=args.api_kind,
        model=args.model_name,
        base_url=os.environ.get(er.OPENAI_BASE_URL_ENV, "").strip() or None,
        api_key_env=er.OPENAI_AUTH_ENV,
    )
    exp = Experiment(
        workspace,
        rollout=graded_rollout,
        reward=er.swebench_reward,
        criterion=valid_when("reward"),
        slice_id="swebench-train",
        instances=train_records,
        seed=er.seed_files(
            model=args.model_name,
            api_kind=args.api_kind,
            base_url=os.environ.get(er.OPENAI_BASE_URL_ENV, "").strip(),
        ),
    )
    strategy = model_program_strategy(
        provider=provider,
        prefix="agent/",
        system_prompt=SYSTEM_PROMPT,
        parent_selection=args.parent_selection,
    )
    components = SimpleNamespace(
        rollout=graded_rollout,
        reward=er.swebench_reward,
        strategy=strategy,
        criterion=valid_when("reward"),
    )

    def announce(decision: Any) -> None:
        print(
            f"outcome={decision.outcome} "
            f"candidate={decision.candidate.get('hash')} "
            f"train_reward={_score(decision.candidate):.4g} reason={decision.reason}"
        )

    open_ended.run_evolution(
        workspace,
        components,
        exp.slice,
        rounds=rounds,
        branches=branches,
        meta_workers=meta_workers,
        on_decision=announce,
    )

    best = best_archive_version(workspace) or exp.current()
    print(f"\nbest-in-archive version: {best.hash}")
    run_heldout_scoring(args, layout, test_records, best, base_rollout)


def resolve_schedule(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return ``(rounds, branches, meta_workers)`` from CLI args.

    ``--rounds`` wins; otherwise the deprecated ``--generations`` sets rounds; the
    fallback is 4. Total candidates evaluated is ``rounds * branches``.
    """

    if args.rounds is not None:
        rounds = args.rounds
    elif args.generations is not None:
        rounds = args.generations
    else:
        rounds = 4
    branches = max(1, int(args.branches))
    meta_workers = int(args.meta_concurrency) or branches
    return max(1, int(rounds)), branches, max(1, meta_workers)


def pick_best_node(
    nodes: Sequence[archive.ArchiveNode], *, dim: str = "reward"
) -> archive.ArchiveNode | None:
    """Return the highest-scoring valid archive node, or None when none qualify."""

    candidates = [node for node in nodes if node.valid_parent and dim in node.scores]
    if not candidates:
        return None
    return max(candidates, key=lambda node: node.scores[dim])


def best_archive_version(workspace: Path, *, dim: str = "reward") -> Version | None:
    """Resolve the best-on-train archive member to a Version, or None.

    DGM reports the best generated agent on the held-out test, not just
    the final promoted pointer; this picks that agent from the decision-log
    archive so the headline is robust to promotion mechanics.
    """

    best = pick_best_node(archive.nodes(workspace), dim=dim)
    if best is None:
        return None
    return evo_store.version(workspace, best.hash)


def print_plan(
    args: argparse.Namespace,
    layout: er.PerformanceLayout,
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> None:
    print(f"run root: {layout.run_root}")
    print(f"evolution workspace: {layout.evolution_workspace}")
    print(f"swebench runs: {layout.swebench_runs}")
    print(f"train dataset: {args.train_dataset}")
    print(f"train records: {len(train_records)}")
    print(f"test dataset: {args.test_dataset}")
    print(f"test records: {len(test_records)}")
    rounds, branches, meta_workers = resolve_schedule(args)
    print(f"rounds: {rounds}")
    print(f"branches: {branches}")
    print(f"meta-concurrency: {meta_workers}")
    print(f"candidates (rounds x branches): {rounds * branches}")
    print(f"parallel: {args.parallel}")
    print(f"parent selection: {args.parent_selection}")
    print(f"model: {args.model_name}")
    print(f"api kind: {args.api_kind}")
    print(f"max turns: {args.max_turns}")


def run_heldout_scoring(
    args: argparse.Namespace,
    layout: er.PerformanceLayout,
    test_records: Sequence[Mapping[str, Any]],
    version: Version,
    base_rollout,
) -> None:
    print("\nheld-out test rollout:")
    test_slice = heldout_slice(version, test_records)
    base_rollout(version, test_slice)
    source_run_id = heldout_run_id(version, test_records)
    print(f"test run id: {source_run_id}")
    collect = er.collect_predictions_command(
        layout,
        dataset_name=args.dataset_name,
        model_name=f"{args.model_name}-{args.parent_selection}",
        source_run_id=source_run_id,
    )
    official = er.official_eval_command(
        layout,
        dataset_name=args.dataset_name,
        instance_ids=er.instance_ids(test_records),
        max_workers=args.parallel,
    )
    print("\ncollect predictions:")
    print(" ".join(collect))
    subprocess.run(collect, cwd=ROOT, check=True)
    print("\nofficial scoring:")
    print(" ".join(official))
    subprocess.run(official, cwd=ROOT, check=True)


def heldout_slice(version: Version, test_records: Sequence[Mapping[str, Any]]) -> Slice:
    del version  # scaffold reaches the container via version_artifacts staging
    return Slice("swebench-test", tuple(dict(rec) for rec in test_records))


def heldout_run_id(version: Version, test_records: Sequence[Mapping[str, Any]]) -> str:
    slice_ = heldout_slice(version, test_records)
    return f"{version.hash}-{slice_.sha}"


def print_monitor(args: argparse.Namespace) -> None:
    _shared.load_dotenv(args.dotenv)
    run_root = Path(args.output_root) / args.run_id
    summary = subprocess.run(
        [
            sys.executable,
            str(ROOT / "recipes" / "dgm" / "report.py"),
            str(run_root),
            "--test-dataset",
            args.test_dataset,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(summary.stdout, end="")
    if summary.returncode != 0:
        print(summary.stderr, file=sys.stderr, end="")
        raise SystemExit(summary.returncode)


def _score(record: Mapping[str, Any]) -> float:
    scores = record.get("scores", {})
    if isinstance(scores, Mapping):
        return float(scores.get("reward", 0.0) or 0.0)
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-name", default=er.DEFAULT_DATASET)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of sequential evolution rounds. Default: 4 (or --generations).",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=3,
        help="Candidate branches evaluated concurrently per round. Default: 3.",
    )
    parser.add_argument(
        "--meta-concurrency",
        type=int,
        default=0,
        help="Concurrent meta-agent LLM calls per round. Default: 0 (= --branches).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Deprecated alias for --rounds (total candidates = rounds x branches).",
    )
    parser.add_argument(
        "--parent-selection",
        choices=["latest", "best", "score_prop", "score_child_prop"],
        default="score_child_prop",
    )
    parser.add_argument(
        "--parallel",
        default=_shared.AUTO_PARALLEL,
        help="Global worker cap, or 'auto' to size to the Docker VM. Default: auto.",
    )
    parser.add_argument(
        "--model-name", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--api-kind",
        choices=["openai-chat", "openai-responses"],
        default="openai-chat",
    )
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--wheelhouse", default="")
    parser.add_argument("--uv-binary", default="")
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.monitor:
        print_monitor(args)
        return
    run_workflow(args)


if __name__ == "__main__":
    main()
