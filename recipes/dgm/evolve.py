"""Faithful DGM self-evolving SWE-bench recipe.

A model-driven meta-agent rewrites the whole agent program under ``agent/``; the
evolution kernel runs a parallel open-ended archive-admission loop (branches per
round, best-valid promotion, archive parent selection), and the best-on-train
agent is scored on a held-out test split. All DGM knobs are exposed. Dry by
default; --execute runs real model + Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for recipe support modules
sys.path.insert(0, str(ROOT / "src"))

import recipes.runtime as recipe_runtime  # noqa: E402
from recipes.dgm import swebench as er  # noqa: E402
from recipes.dgm.algorithm import archive, open_ended  # noqa: E402
from simple_agent_lab.evolution import Experiment  # noqa: E402
from simple_agent_lab.evolution.components.criterion import valid_when  # noqa: E402
from simple_agent_lab.evolution.components.strategy import (  # noqa: E402
    model_program_strategy,
)
from simple_agent_lab.evolution.kernel import store as evo_store  # noqa: E402
from simple_agent_lab.evolution.config import safe_run_root  # noqa: E402
from simple_agent_lab.evolution.types import RunScores, Slice, Verdict, Version  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402
from simple_agent_lab.trace.jsonl import read_jsonl  # noqa: E402

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
    recipe_runtime.load_dotenv(args.dotenv)
    if args.model_name == DEFAULT_MODEL and os.environ.get("OPENAI_MODEL"):
        args.model_name = os.environ["OPENAI_MODEL"]
    output_root = Path(args.output_root)
    run_root = safe_run_root(output_root, args.run_id)
    workspace = run_root / "evolution"
    if args.reset and run_root.exists():
        recipe_runtime.cleanup_reset_containers(run_root)
        shutil.rmtree(run_root)
    layout = er.PerformanceLayout(output_root, args.run_id)
    layout.create()

    train_records = er.load_dataset(args.train_dataset)
    test_records = er.load_dataset(args.test_dataset)
    print_plan(args, layout, train_records, test_records)
    if not args.execute:
        print("\ndry run only; pass --execute to run model + Docker evolution")
        return

    recipe_runtime.check_docker_available()
    rounds, branches, meta_workers = resolve_schedule(args)
    resolution = recipe_runtime.resolve_parallel_workers(
        args.parallel, len(train_records)
    )
    global_workers = resolution.workers
    per_branch = recipe_runtime.branch_concurrency(
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
        criterion=dgm_admission_criterion("reward"),
        slice_id="swebench-train",
        instances=train_records,
        seed=er.seed_files(
            model=args.model_name,
            api_kind=args.api_kind,
            base_url=os.environ.get(er.OPENAI_BASE_URL_ENV, "").strip(),
        ),
    )
    baseline_test = run_heldout_scoring(
        args,
        layout,
        test_records,
        exp.current(),
        base_rollout,
        label="baseline",
        record_generation=False,
    )
    strategy = model_program_strategy(
        provider=provider,
        prefix="agent/",
        system_prompt=SYSTEM_PROMPT,
        parent_selection=args.parent_selection,
        parent_selector=select_archive_parent,
    )
    components = SimpleNamespace(
        rollout=graded_rollout,
        reward=er.swebench_reward,
        strategy=strategy,
        criterion=dgm_admission_criterion("reward"),
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
    final_test = run_heldout_scoring(
        args,
        layout,
        test_records,
        best,
        base_rollout,
        label="final",
        record_generation=True,
    )
    summary_path = write_test_summary(layout, baseline=baseline_test, final=final_test)
    print_test_summary(summary_path)


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


def select_archive_parent(ctx: Any, method: str) -> str:
    nodes = archive.nodes(ctx.workspace)
    if not nodes:
        return ctx.current.hash
    try:
        return archive.select_parent(nodes, method=method)
    except ValueError:
        return ctx.current.hash


def dgm_admission_criterion(dim: str = "reward"):
    """Admit worse-but-valid children, but reject evolved-agent fallback runs."""

    base_criterion = valid_when(dim)

    def judge(baseline: RunScores, candidate: RunScores) -> Verdict:
        verdict = base_criterion(baseline, candidate)
        if any(float(scores.get(dim, 0.0)) < 0.0 for scores in candidate.values()):
            deltas = dict(verdict.deltas)
            deltas["valid_parent"] = 0.0
            return Verdict(
                False,
                f"admission invalid (negative {dim}; likely evolved-agent fallback)",
                deltas,
            )
        return verdict

    return judge


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
    *,
    label: str,
    record_generation: bool = True,
) -> dict[str, Any]:
    artifacts = er.official_artifacts(layout, label)
    print(f"\nheld-out test rollout ({label}):")
    test_slice = heldout_slice(version, test_records)
    base_rollout(version, test_slice)
    source_run_id = heldout_run_id(version, test_records)
    print(f"test run id: {source_run_id}")
    collect = er.collect_predictions_command(
        layout,
        dataset_name=args.dataset_name,
        model_name=f"{args.model_name}-{args.parent_selection}",
        source_run_id=source_run_id,
        predictions=artifacts.predictions,
    )
    official = er.official_eval_command(
        layout,
        dataset_name=args.dataset_name,
        instance_ids=er.instance_ids(test_records),
        max_workers=args.parallel,
        predictions=artifacts.predictions,
        eval_results=artifacts.eval_results,
        official_output_dir=artifacts.harness,
        run_id=artifacts.run_id,
    )
    print("\ncollect predictions:")
    print(" ".join(collect))
    subprocess.run(collect, cwd=ROOT, check=True)
    print("\nofficial scoring:")
    print(" ".join(official))
    subprocess.run(official, cwd=ROOT, check=True)
    summary = er.summarize_official_eval_results(artifacts.eval_results)
    record: dict[str, Any] = {
        "label": label,
        "version": version.hash,
        "parent": version.parent or "",
        "source_run_id": source_run_id,
        "predictions": str(artifacts.predictions),
        "eval_results": str(artifacts.eval_results),
        **summary,
    }
    print_score_line(label, record)
    if record_generation:
        record_heldout_generation(
            layout,
            version,
            parent_selection=args.parent_selection,
            eval_results=artifacts.eval_results,
        )
    return record


def record_heldout_generation(
    layout: er.PerformanceLayout,
    version: Version,
    *,
    parent_selection: str,
    eval_results: str | Path | None = None,
) -> dict[str, Any]:
    """Write one ``generation_metrics`` row from the held-out official eval results.

    This is the wiring that makes ``report.summarize`` real: the official
    scoring step writes per-instance rows to ``official/eval_results.jsonl``;
    here we aggregate them into a single recipe-level summary row capturing the
    best version's held-out official resolved rate. Docker-free and unit-testable.
    """

    eval_results_path = (
        Path(eval_results) if eval_results is not None else layout.eval_results
    )
    if not eval_results_path.is_file():
        return {}
    rows = read_jsonl(eval_results_path)
    if not rows:
        return {}
    runs = [
        {
            "reward": float(row.get("score", 0.0) or 0.0),
            "patch_chars": int((row.get("metrics") or {}).get("patch_chars", 0) or 0),
            "tokens": 0,
        }
        for row in rows
    ]
    record = er.generation_metric_record(
        generation=0,
        version_hash=version.hash,
        parent_hash=version.parent or "",
        parent_selection=parent_selection,
        decision_outcome="heldout",
        runs=runs,
    )
    record["test_resolved_rate"] = record["resolved_rate"]
    er.write_generation_metrics(layout.generation_metrics, [record])
    return record


def write_test_summary(
    layout: er.PerformanceLayout,
    *,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> Path:
    baseline_resolved = _as_int(baseline.get("resolved"))
    final_resolved = _as_int(final.get("resolved"))
    baseline_rate = _as_float(baseline.get("resolved_rate"))
    final_rate = _as_float(final.get("resolved_rate"))
    summary = {
        "baseline": dict(baseline),
        "final": dict(final),
        "delta_resolved": final_resolved - baseline_resolved,
        "delta_resolved_rate": final_rate - baseline_rate,
    }
    path = layout.run_root / "test_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def print_test_summary(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    print("\nheld-out test summary:")
    print_score_line("baseline", summary["baseline"])
    print_score_line("final", summary["final"])
    delta_resolved = int(summary.get("delta_resolved", 0))
    delta_rate = float(summary.get("delta_resolved_rate", 0.0))
    print(f"delta: {delta_resolved:+d} / {delta_rate:+.3f}")
    print(f"summary: {path}")


def print_score_line(label: str, record: Mapping[str, Any]) -> None:
    resolved = _as_int(record.get("resolved"))
    total = _as_int(record.get("total"))
    rate = _as_float(record.get("resolved_rate"))
    print(f"{label} test: {resolved}/{total} = {rate:.3f}")


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value:
        return int(value)
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        return float(value)
    return 0.0


def heldout_slice(version: Version, test_records: Sequence[Mapping[str, Any]]) -> Slice:
    del version  # scaffold reaches the container via version_artifacts staging
    return Slice("swebench-test", tuple(dict(rec) for rec in test_records))


def heldout_run_id(version: Version, test_records: Sequence[Mapping[str, Any]]) -> str:
    slice_ = heldout_slice(version, test_records)
    return f"{version.hash}-{slice_.sha}"


def print_monitor(args: argparse.Namespace) -> None:
    recipe_runtime.load_dotenv(args.dotenv)
    run_root = safe_run_root(args.output_root, args.run_id)
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
        default=recipe_runtime.AUTO_PARALLEL,
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
