"""Run the real model + Docker SWE-bench self-evolving workflow.

This is the production-shaped default: a SAL meta-agent evolves text playbook
artifacts, the evolution kernel compares candidates on the train dataset, and the
SWE-bench Docker rollout evaluates each version.
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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evolution_recipes.hyperagents import (  # noqa: E402
    open_ended,
    sal_meta_strategy,
    swebench_driver,
    whole_repo_strategy,
)
from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY  # noqa: E402
from simple_agent_lab.evals.suites.swebench import agent_package as agent_pkg  # noqa: E402
from simple_agent_lab.evolution import Experiment, archive  # noqa: E402
from simple_agent_lab.evolution.components.criterion import valid_when  # noqa: E402
from simple_agent_lab.evolution.kernel import store as evo_store  # noqa: E402
from simple_agent_lab.evolution.types import Run, Slice, Version  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("evals/out/self_evolving/swebench")
DEFAULT_MODEL = "hyperagents-swebench"
AGENT_PREFIX = "agent/"
EVOLVING_CONTAINER_MODULE = "simple_agent_lab.evals.suites.swebench.evolving"


def load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def check_docker_available(*, client_factory=None) -> None:
    """Fail early when Docker is not reachable for SWE-bench execution."""

    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        client = client_factory()
        ping = getattr(client, "ping", None)
        if callable(ping):
            ping()
    except Exception as exc:
        raise SystemExit(
            "Docker is required for --execute SWE-bench self-evolution, but the "
            "Docker daemon is not reachable. Start Docker Desktop or Colima and "
            "set DOCKER_HOST if needed. On macOS/Colima, try: "
            "colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta "
            "&& export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def cleanup_reset_containers(run_root: str | Path, *, client_factory=None) -> int:
    """Remove stale SWE-bench Docker containers owned by an existing run root."""

    swebench_runs = Path(run_root) / "swebench_runs"
    if not swebench_runs.is_dir():
        return 0
    rollout_ids = sorted(path.name for path in swebench_runs.iterdir() if path.is_dir())
    if not rollout_ids:
        return 0

    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        client = client_factory()
    except Exception:
        return 0

    removed = 0
    for rollout_id in rollout_ids:
        try:
            containers = client.containers.list(all=True, filters={"name": rollout_id})
        except Exception:
            continue
        for container in containers:
            name = str(getattr(container, "name", ""))
            if not (name.startswith("swebench.") and name.endswith(f".{rollout_id}")):
                continue
            try:
                container.remove(force=True)
            except Exception:
                continue
            removed += 1
    return removed


def ensure_rollout_artifacts(runs: Sequence[Run]) -> None:
    if not runs:
        raise RuntimeError("SWE-bench rollout produced no run directories.")
    missing = [str(run.dir / "out" / "result.json") for run in runs if not run.ok]
    if missing:
        raise RuntimeError(
            "SWE-bench rollout missing result.json artifacts. This usually means "
            "Docker/container execution failed before the agent produced a patch. "
            "Missing: " + ", ".join(missing)
        )


def seed_files(*, model: str, api_kind: str, base_url: str = "") -> dict[str, str]:
    provider: dict[str, Any] = {
        "api": api_kind,
        "model": model,
        "api_key_env": sal_meta_strategy.OPENAI_AUTH_ENV,
    }
    if base_url:
        provider["base_url"] = base_url
    files: dict[str, str] = {
        "README.md": "# Real SWE-bench self-evolving agent\n",
        "provider.json": json.dumps(provider, indent=2, sort_keys=True) + "\n",
    }
    for name, text in agent_pkg.default_agent_package().items():
        files[AGENT_PREFIX + name] = text
    return files


def package_files(version: Version) -> dict[str, str]:
    """Collect the version's evolvable agent package, or the neutral default."""

    out: dict[str, str] = {}
    for name in version.files():
        if name.startswith(AGENT_PREFIX):
            out[name[len(AGENT_PREFIX) :]] = version.read(name)
    return out or agent_pkg.default_agent_package()


def version_package_artifacts(version: Version) -> dict[str, bytes]:
    """Stage the version's package under AGENT_PACKAGE_KEY for the in-container agent."""

    payload = json.dumps(package_files(version), ensure_ascii=False)
    return {AGENT_PACKAGE_KEY: payload.encode("utf-8")}


def reward_from_result(result: Mapping[str, Any]) -> float:
    if "resolved" in result:
        return 1.0 if bool(result.get("resolved")) else 0.0
    if "score" in result:
        return float(result.get("score") or 0.0)
    value = result.get("reward", 0.0)
    return float(value or 0.0)


def swebench_reward(run: Run) -> float:
    return reward_from_result(run.result)


def apply_eval_score(run: Run, eval_row: Mapping[str, Any]) -> None:
    path = run.dir / "out" / "result.json"
    result = dict(run.result)
    metrics = eval_row.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    score = float(eval_row.get("score", 0.0) or 0.0)
    result.update(
        {
            "resolved": bool(metrics.get("resolved") or eval_row.get("passed")),
            "status": str(metrics.get("status") or eval_row.get("reason") or ""),
            "score": score,
            "reward": score,
        }
    )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def grade_reuse_runs(
    runs: Sequence[Run],
    instances: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    model_name: str,
) -> None:
    from evals.swebench.evaluate_predictions import reuse_eval_row

    by_id = {str(instance.get("instance_id")): instance for instance in instances}
    for run in runs:
        instance = by_id.get(run.instance_id)
        if instance is None:
            continue
        row = reuse_eval_row(
            instance, run.result, dataset_name=dataset_name, model_name=model_name
        )
        apply_eval_score(run, row)


def make_scaffold_rollout(base_rollout, *, dataset_name: str, model_name: str):
    """Wrap the base rollout to enforce artifacts and grade with official reuse.

    The evolved agent package reaches the container via ``version_artifacts``
    staging on the base rollout, so instances are passed through unchanged here.
    """

    def rollout(version: Version, slice_: Slice) -> Sequence[Run]:
        runs = base_rollout(version, slice_)
        ensure_rollout_artifacts(runs)
        grade_reuse_runs(
            runs, slice_.instances, dataset_name=dataset_name, model_name=model_name
        )
        return runs

    return rollout


def run_workflow(args: argparse.Namespace) -> None:
    load_dotenv(args.dotenv)
    if args.model_name == DEFAULT_MODEL and os.environ.get("OPENAI_MODEL"):
        args.model_name = os.environ["OPENAI_MODEL"]
    output_root = Path(args.output_root)
    run_root = output_root / args.run_id
    workspace = run_root / "evolution"
    if args.reset and run_root.exists():
        cleanup_reset_containers(run_root)
        shutil.rmtree(run_root)
    layout = swebench_driver.PerformanceLayout(output_root, args.run_id)
    layout.create()

    train_records = swebench_driver.load_dataset(args.train_dataset)
    test_records = swebench_driver.load_dataset(args.test_dataset)
    print_plan(args, layout, train_records, test_records)
    if not args.execute:
        print("\ndry run only; pass --execute to run model + Docker evolution")
        return

    check_docker_available()
    rounds, branches, meta_workers = resolve_schedule(args)
    resolution = swebench_driver.resolve_parallel_workers(
        args.parallel, len(train_records)
    )
    global_workers = resolution.workers
    per_branch = swebench_driver.branch_concurrency(
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
    base_rollout = swebench_driver.build_swebench_rollout(
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
        version_artifacts=version_package_artifacts,
        container_module=EVOLVING_CONTAINER_MODULE,
    )
    graded_rollout = make_scaffold_rollout(
        base_rollout,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    provider = Provider(
        id="openai-chat",
        api=args.api_kind,
        model=args.model_name,
        base_url=os.environ.get(sal_meta_strategy.OPENAI_BASE_URL_ENV, "").strip()
        or None,
        api_key_env=sal_meta_strategy.OPENAI_AUTH_ENV,
    )
    exp = Experiment(
        workspace,
        rollout=graded_rollout,
        reward=swebench_reward,
        criterion=valid_when("reward"),
        slice_id="swebench-train",
        instances=train_records,
        seed=seed_files(
            model=args.model_name,
            api_kind=args.api_kind,
            base_url=os.environ.get(sal_meta_strategy.OPENAI_BASE_URL_ENV, "").strip(),
        ),
    )
    strategy = whole_repo_strategy.make_strategy(
        workspace,
        provider=provider,
        parent_selection=args.parent_selection,
    )
    components = SimpleNamespace(
        rollout=graded_rollout,
        reward=swebench_reward,
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

    candidates = [
        node for node in nodes if node.valid_parent and dim in node.scores
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda node: node.scores[dim])


def best_archive_version(
    workspace: Path, *, dim: str = "reward"
) -> Version | None:
    """Resolve the best-on-train archive member to a Version, or None.

    HyperAgents reports the best generated agent on the held-out test, not just
    the final promoted pointer; this picks that agent from the decision-log
    archive so the headline is robust to promotion mechanics.
    """

    best = pick_best_node(archive.nodes(workspace), dim=dim)
    if best is None:
        return None
    return evo_store.version(workspace, best.hash)


def print_plan(
    args: argparse.Namespace,
    layout: swebench_driver.PerformanceLayout,
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
    layout: swebench_driver.PerformanceLayout,
    test_records: Sequence[Mapping[str, Any]],
    version: Version,
    base_rollout,
) -> None:
    print("\nheld-out test rollout:")
    test_slice = heldout_slice(version, test_records)
    base_rollout(version, test_slice)
    source_run_id = heldout_run_id(version, test_records)
    print(f"test run id: {source_run_id}")
    collect = swebench_driver.collect_predictions_command(
        layout,
        dataset_name=args.dataset_name,
        model_name=f"{args.model_name}-{args.parent_selection}",
        source_run_id=source_run_id,
    )
    official = swebench_driver.official_eval_command(
        layout,
        dataset_name=args.dataset_name,
        instance_ids=swebench_driver.instance_ids(test_records),
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
    load_dotenv(args.dotenv)
    run_root = Path(args.output_root) / args.run_id
    summary = subprocess.run(
        [
            sys.executable,
            "scripts/evolution_recipes/hyperagents/report_swebench.py",
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
    parser.add_argument("--dataset-name", default=swebench_driver.DEFAULT_DATASET)
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
        default=swebench_driver.AUTO_PARALLEL,
        help="Global worker cap, or 'auto' to size to the Docker VM. Default: auto.",
    )
    parser.add_argument(
        "--model-name", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument("--api-kind", default="openai-chat")
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
