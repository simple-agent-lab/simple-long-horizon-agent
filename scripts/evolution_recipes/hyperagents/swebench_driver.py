"""SWE-bench performance driver for the HyperAgents recipe.

The real benchmark work stays on the mature SWE-bench path. This module owns the
recipe-specific bookkeeping: output layout, archive/evolution metadata, and the
commands that bridge generic `result.json` runs into official SWE-bench scoring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simple_agent_lab.trace.jsonl import write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("evals/out/hyperagents_swebench")
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MODEL_NAME = "hyperagents-swebench"

# --parallel auto-sizing. The SWE-bench agent loop is latency-bound on the remote
# model, so the local ceiling is the Docker VM, not the host: each container peaks
# at roughly PER_CONTAINER_GB during test bursts (astropy/sympy/matplotlib run
# pytest), and overrunning the VM's RAM lets its OOM killer drop containers mid-run.
# So "auto" fits workers under the VM's memory (HARD cap — exceeding it triggers
# the OOM killer) and oversubscribes its CPUs (SOFT cap — most turns are idle model
# waits or sub-second bash, so extra threads just queue, they don't crash). Memory
# is therefore never oversubscribed; CPU is, since the machine is otherwise idle.
AUTO_PARALLEL = "auto"
PER_CONTAINER_GB = 1.5
DOCKER_MEMORY_RESERVE_GB = 1.5
CPU_OVERSUBSCRIBE = 3
FALLBACK_PARALLEL = 2


@dataclass(frozen=True)
class ParallelResolution:
    """A resolved worker count plus a human-readable reason for the plan output."""

    workers: int
    detail: str


def _docker_resources(client_factory: Any = None) -> tuple[int, float] | None:
    """Return ``(NCPU, MemTotal_GB)`` for the Docker daemon, or None if unknown."""

    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        info = client_factory().info()
        ncpu = int(info.get("NCPU") or 0)
        mem_gb = int(info.get("MemTotal") or 0) / (1024**3)
    except Exception:
        return None
    if ncpu <= 0 or mem_gb <= 0:
        return None
    return ncpu, mem_gb


def resolve_parallel_workers(
    requested: str | int,
    num_instances: int,
    *,
    client_factory: Any = None,
) -> ParallelResolution:
    """Resolve ``--parallel``, auto-sizing to the Docker VM when ``"auto"``.

    An explicit positive integer is honored as-is. ``"auto"`` returns the largest
    worker count that fits the VM's memory and CPU caps without exceeding the slice
    size; when ``docker info`` is unreachable it falls back to a small safe count.
    """

    instances = max(1, int(num_instances))
    text = str(requested).strip().lower()
    if text != AUTO_PARALLEL:
        try:
            explicit = int(text)
        except ValueError:
            raise SystemExit(
                f"--parallel must be a positive integer or 'auto'; got {requested!r}"
            )
        if explicit < 1:
            raise SystemExit(
                f"--parallel must be >= 1 or 'auto'; got {requested!r}"
            )
        return ParallelResolution(explicit, f"explicit; {instances} instances")

    resources = _docker_resources(client_factory)
    if resources is None:
        return ParallelResolution(
            min(instances, FALLBACK_PARALLEL),
            "auto fallback (docker info unavailable)",
        )
    ncpu, mem_gb = resources
    mem_cap = max(1, int((mem_gb - DOCKER_MEMORY_RESERVE_GB) // PER_CONTAINER_GB))
    cpu_cap = ncpu * CPU_OVERSUBSCRIBE
    workers = max(1, min(instances, mem_cap, cpu_cap))
    detail = (
        f"auto: docker VM {ncpu} cpu / {mem_gb:.1f} GB; "
        f"mem cap {mem_cap} (@{PER_CONTAINER_GB:g} GB/container), "
        f"cpu cap {cpu_cap}, {instances} instances"
    )
    return ParallelResolution(workers, detail)


def branch_concurrency(*, global_workers: int, branches: int) -> int:
    """Per-branch Docker concurrency so all branches stay within the global cap.

    Two parallelism levels (instances within a rollout x branches per round)
    share one Docker VM. ``global_workers`` is the memory-bounded hard cap from
    ``resolve_parallel_workers``; splitting it across branches keeps in-flight
    containers <= ``global_workers`` (the stability guarantee), with a floor of 1
    so every branch always makes progress.
    """

    return max(1, int(global_workers) // max(1, int(branches)))


@dataclass(frozen=True)
class PerformanceLayout:
    output_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.output_root / self.run_id

    @property
    def evolution_workspace(self) -> Path:
        return self.run_root / "evolution"

    @property
    def swebench_runs(self) -> Path:
        return self.run_root / "swebench_runs"

    @property
    def official(self) -> Path:
        return self.run_root / "official"

    @property
    def predictions(self) -> Path:
        return self.official / f"{self.run_id}_predictions.jsonl"

    @property
    def eval_results(self) -> Path:
        return self.official / "eval_results.jsonl"

    @property
    def generation_metrics(self) -> Path:
        return self.run_root / "generation_metrics.jsonl"

    def create(self) -> None:
        for path in (
            self.evolution_workspace,
            self.swebench_runs,
            self.official,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load full SWE-bench instance records from a JSONL dataset file."""

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def instance_ids(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(record.get("instance_id", "")) for record in records)


def collect_predictions_command(
    layout: PerformanceLayout,
    *,
    dataset_name: str,
    model_name: str,
    source_run_id: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "evals/swebench/evaluate_predictions.py",
        "--collect-predictions",
        "--run-root",
        str(layout.swebench_runs),
        "--run-id",
        source_run_id or layout.run_id,
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--predictions",
        str(layout.predictions),
    ]


def official_eval_command(
    layout: PerformanceLayout,
    *,
    dataset_name: str,
    instance_ids: Sequence[str] = (),
    max_workers: int = 1,
) -> list[str]:
    command = [
        sys.executable,
        "evals/swebench/evaluate_predictions.py",
        "--run-official",
        "--dataset-name",
        dataset_name,
        "--predictions",
        str(layout.predictions),
        "--jsonl",
        str(layout.eval_results),
        "--official-output-dir",
        str(layout.official / "harness"),
        "--run-id",
        layout.run_id,
        "--max-workers",
        str(max_workers),
    ]
    if instance_ids:
        command.extend(["--instance-ids", *instance_ids])
    return command


def ablation_commands(
    *,
    run_id_prefix: str,
    train_dataset: str | Path,
    test_dataset: str | Path,
    selectors: Sequence[str] = ("latest", "best", "score_child_prop"),
    generations: int = 3,
    parallel: int = 1,
) -> list[list[str]]:
    commands = []
    for selector in selectors:
        command = [
            "bash",
            "runs/run_hyperagents_swebench.sh",
            "--run-id",
            f"{run_id_prefix}-{selector.replace('_', '-')}",
            "--train-dataset",
            str(train_dataset),
            "--test-dataset",
            str(test_dataset),
            "--generations",
            str(generations),
            "--parent-selection",
            selector,
            "--parallel",
            str(parallel),
        ]
        commands.append(command)
    return commands


def generation_metric_record(
    *,
    generation: int,
    version_hash: str,
    parent_hash: str,
    parent_selection: str,
    decision_outcome: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(runs)
    resolved = sum(1 for run in runs if float(run.get("reward", 0.0)) > 0.0)
    patch_valid = sum(1 for run in runs if int(run.get("patch_chars", 0)) > 0)
    tokens = sum(int(run.get("tokens", 0)) for run in runs)
    return {
        "generation": generation,
        "version": version_hash,
        "parent": parent_hash,
        "parent_selection": parent_selection,
        "outcome": decision_outcome,
        "total": total,
        "resolved": resolved,
        "resolved_rate": resolved / total if total else 0.0,
        "patch_valid": patch_valid,
        "patch_valid_rate": patch_valid / total if total else 0.0,
        "tokens": tokens,
    }


def build_swebench_rollout(
    layout: PerformanceLayout,
    *,
    dataset_name: str = DEFAULT_DATASET,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
    wheelhouse: str | Path | None = None,
    uv_binary: str | Path | None = None,
    in_env_scoring: bool = False,
    version_artifacts: Any = None,
    container_module: str | None = None,
):
    """Build the evolution rollout on the mature SWE-bench Suite path.

    ``version_artifacts`` is forwarded to ``dataset_rollout`` so a version's
    evolved files (e.g. the agent package) are staged into each run.
    ``container_module`` overrides the suite's container half (e.g. the
    ``evolving`` module that builds the agent from the staged package).
    """

    from evals.swebench.harness import DEFAULT_WHEELHOUSE_MOUNT
    from evals.swebench.suite import SwebenchSuite
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evolution.components.rollout import dataset_rollout

    extra_kwargs = dict(run_kwargs or {})
    if wheelhouse and "wheelhouse_mount" not in extra_kwargs:
        extra_kwargs["wheelhouse_mount"] = DEFAULT_WHEELHOUSE_MOUNT
    suite = SwebenchSuite(dataset_name=dataset_name, in_env_scoring=in_env_scoring)
    if container_module:
        suite.container_module = container_module
    return dataset_rollout(
        suite=suite,
        backend=LocalDockerBackend(wheelhouse=wheelhouse, uv_binary=uv_binary),
        store=LocalDirStore(layout.swebench_runs),
        runs_root=layout.swebench_runs,
        concurrency=concurrency,
        run_kwargs=extra_kwargs,
        version_artifacts=version_artifacts,
    )


def write_generation_metrics(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    write_jsonl(path, records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--instances", nargs="*", default=())
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument(
        "--parent-selection",
        choices=["latest", "best", "score_prop", "score_child_prop"],
        default="score_child_prop",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run collection/official scoring commands. Without this, print the plan.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    layout = PerformanceLayout(Path(args.output_root), args.run_id)
    layout.create()

    print(f"run root: {layout.run_root}")
    print(f"evolution workspace: {layout.evolution_workspace}")
    print(f"swebench runs: {layout.swebench_runs}")
    print(f"parent selection: {args.parent_selection}")
    print(f"generations: {args.generations}")
    train_records = load_dataset(args.train_dataset)
    test_records = load_dataset(args.test_dataset)
    test_scope = tuple(args.instances) or instance_ids(test_records)
    print(f"train dataset: {args.train_dataset}")
    print(f"train records: {len(train_records)}")
    print(f"test dataset: {args.test_dataset}")
    print(f"test records: {len(test_records)}")

    collect = collect_predictions_command(
        layout,
        dataset_name=args.dataset_name,
        model_name=f"{args.model_name}-{args.parent_selection}",
    )
    official = official_eval_command(
        layout,
        dataset_name=args.dataset_name,
        instance_ids=test_scope,
        max_workers=args.parallel,
    )
    print("\ncollect predictions:")
    print(" ".join(collect))
    print("\nofficial scoring:")
    print(" ".join(official))

    if args.execute:
        subprocess.run(collect, cwd=ROOT, check=True)
        subprocess.run(official, cwd=ROOT, check=True)
    else:
        print("\ndry run only; pass --execute after SWE-bench runs exist")


if __name__ == "__main__":
    main()
