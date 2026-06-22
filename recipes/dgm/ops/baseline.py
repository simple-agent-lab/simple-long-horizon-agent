"""Measure the seed agent's per-instance resolve, then build a headroom split.

The current splits are too easy (base already ~75%) and too small to show signal.
This script rolls the seed scaffold once over a candidate pool, records which
instances the base resolves, and selects a balanced "headroom" set (a mix of
passing and failing instances so the base resolve rate lands in a detectable
band) split into disjoint train/test JSONL files. The pool can come from a JSONL
file or be fetched from SWE-bench Verified and down-selected by repository.

Selection (``select_headroom``) and splitting (``split_chosen``) are pure and
unit-tested; the Docker measurement pass reuses the same SWE-bench rollout the
evolution runner uses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # for recipe support modules
sys.path.insert(0, str(ROOT / "src"))

import recipes.runtime as recipe_runtime  # noqa: E402
from evals.swebench import evolution_adapter as er  # noqa: E402
from simple_agent_lab.evolution.run_paths import safe_run_root  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("evals/out/dgm_swebench")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps(dict(rec), ensure_ascii=False, sort_keys=True) + "\n"
            for rec in records
        ),
        encoding="utf-8",
    )


def repo_key(record: Mapping[str, Any]) -> str:
    repo = str(record.get("repo") or "").strip()
    if repo:
        return repo
    instance_id = str(record.get("instance_id") or "")
    if "__" in instance_id:
        return instance_id.split("__", 1)[0]
    return "unknown"


def select_diverse_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    want: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Pick a repo-balanced candidate pool with deterministic shuffling."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in pool:
        groups.setdefault(repo_key(record), []).append(dict(record))
    rng = random.Random(seed)
    for records in groups.values():
        rng.shuffle(records)
    repos = sorted(groups)
    rng.shuffle(repos)

    chosen: list[dict[str, Any]] = []
    target = min(max(0, int(want)), len(pool))
    while len(chosen) < target and any(groups[repo] for repo in repos):
        for repo in repos:
            if groups[repo]:
                chosen.append(groups[repo].pop())
                if len(chosen) >= target:
                    break
    return chosen


def load_dataset_pool(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def select_headroom(
    rows: Sequence[Mapping[str, Any]],
    *,
    want: int,
    pass_fraction: float = 0.45,
    seed: int = 0,
) -> list[str]:
    """Pick ``want`` instance ids mixing passes and fails for detectable headroom.

    ``rows`` are baseline records ``{instance_id, resolved}``. Aims for roughly
    ``pass_fraction`` resolved so the base rate lands mid-band; backfills from
    whichever class is larger when one is short.
    """

    passes = [str(r["instance_id"]) for r in rows if r.get("resolved")]
    fails = [str(r["instance_id"]) for r in rows if not r.get("resolved")]
    rng = random.Random(seed)
    rng.shuffle(passes)
    rng.shuffle(fails)

    want = min(want, len(passes) + len(fails))
    want_pass = min(len(passes), round(want * pass_fraction))
    want_fail = min(len(fails), want - want_pass)
    chosen = passes[:want_pass] + fails[:want_fail]

    remainder = passes[want_pass:] + fails[want_fail:]
    rng.shuffle(remainder)
    while len(chosen) < want and remainder:
        chosen.append(remainder.pop())
    return chosen


def split_chosen(
    chosen: Sequence[str],
    pool: Sequence[Mapping[str, Any]],
    *,
    train_size: int,
    test_size: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split chosen ids into disjoint train/test full instance records."""

    by_id = {str(rec.get("instance_id")): dict(rec) for rec in pool}
    ids = [cid for cid in chosen if cid in by_id]
    rng = random.Random(seed)
    rng.shuffle(ids)
    total = train_size + test_size
    if len(ids) < total:
        raise ValueError(f"need {total} headroom instances; only {len(ids)} available")
    train_ids = ids[:train_size]
    test_ids = ids[train_size:total]
    return (
        [by_id[i] for i in train_ids],
        [by_id[i] for i in test_ids],
    )


def measure_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    output_root: Path,
    dataset_name: str,
    concurrency: int,
    api_kind: str,
    max_turns: int,
    model_name: str,
    base_url: str = "",
    wheelhouse: str = "",
    uv_binary: str = "",
) -> list[dict[str, Any]]:
    """Roll the seed scaffold over ``pool`` once and return resolve records."""

    from simple_agent_lab.evolution.kernel import store as evo_store
    from simple_agent_lab.evolution.types import Slice

    safe_run_root(output_root, run_id)
    layout = er.PerformanceLayout(output_root, run_id)
    layout.create()
    workspace = layout.run_root / "evolution"

    seed = evo_store.stage(
        workspace,
        base=None,
        edits=er.seed_files(model=model_name, api_kind=api_kind, base_url=base_url),
    )
    evo_store.promote(workspace, seed)

    base_rollout = er.build_swebench_rollout(
        layout,
        dataset_name=dataset_name,
        concurrency=concurrency,
        run_kwargs={"api_kind": api_kind, "max_turns": max_turns},
        wheelhouse=wheelhouse or None,
        uv_binary=uv_binary or None,
        in_env_scoring=True,
        version_artifacts=er.version_package_artifacts,
        container_module=er.EVOLVING_CONTAINER_MODULE,
    )
    rollout = er.make_scaffold_rollout(
        base_rollout, dataset_name=dataset_name, model_name=model_name
    )
    runs = rollout(seed, Slice("baseline-pool", tuple(dict(r) for r in pool)))
    return [
        {
            "instance_id": run.instance_id,
            "resolved": er.reward_from_result(run.result) > 0.0,
            "score": er.reward_from_result(run.result),
        }
        for run in runs
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        default="",
        help="Candidate pool JSONL. When omitted, fetch --dataset-name/--dataset-split.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="Repo-balanced pool size. Default: all provided pool, or 160 when fetching.",
    )
    parser.add_argument(
        "--pool-out",
        default="",
        help="Optional path to write the selected diverse pool JSONL.",
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--pass-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="baseline-pool")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-name", default=er.DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--parallel", default=recipe_runtime.AUTO_PARALLEL)
    parser.add_argument(
        "--model-name",
        default=os.environ.get("OPENAI_MODEL", ""),
        help="Provider model; default $OPENAI_MODEL.",
    )
    parser.add_argument("--api-kind", default="openai-chat")
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument("--wheelhouse", default="")
    parser.add_argument("--uv-binary", default="")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument(
        "--baseline-out",
        default="",
        help="Optional path to write the per-instance baseline_resolve.jsonl.",
    )
    parser.add_argument(
        "--reuse-baseline",
        default="",
        help="Skip the Docker pass; read an existing baseline_resolve.jsonl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    safe_run_root(args.output_root, args.run_id)

    recipe_runtime.load_dotenv(args.dotenv)
    if args.pool:
        pool = read_jsonl(args.pool)
        pool_size = args.pool_size
    else:
        pool = load_dataset_pool(args.dataset_name, args.dataset_split)
        pool_size = args.pool_size or 160
    if pool_size:
        pool = select_diverse_pool(pool, want=pool_size, seed=args.seed)
    if args.pool_out:
        write_jsonl(args.pool_out, pool)

    model_name = args.model_name or os.environ.get("OPENAI_MODEL", "")
    base_url = os.environ.get(er.OPENAI_BASE_URL_ENV, "").strip()
    pool_ids = {str(record.get("instance_id") or "") for record in pool}

    if args.reuse_baseline:
        baseline = [
            row
            for row in read_jsonl(args.reuse_baseline)
            if str(row.get("instance_id") or "") in pool_ids
        ]
    else:
        if not model_name:
            raise SystemExit(
                "No model: pass --model-name or set OPENAI_MODEL in the env/.env."
            )
        recipe_runtime.check_docker_available()
        resolution = recipe_runtime.resolve_parallel_workers(args.parallel, len(pool))
        print(f"baseline pass workers: {resolution.workers} ({resolution.detail})")
        print(f"provider model: {model_name}  base_url: {base_url or '(default)'}")
        baseline = measure_pool(
            pool,
            run_id=args.run_id,
            output_root=Path(args.output_root),
            dataset_name=args.dataset_name,
            concurrency=resolution.workers,
            api_kind=args.api_kind,
            max_turns=args.max_turns,
            model_name=model_name,
            base_url=base_url,
            wheelhouse=args.wheelhouse,
            uv_binary=args.uv_binary,
        )

    if args.baseline_out:
        write_jsonl(args.baseline_out, baseline)
    resolved = sum(1 for r in baseline if r.get("resolved"))
    print(f"baseline: {resolved}/{len(baseline)} resolved")

    chosen = select_headroom(
        baseline,
        want=args.train_size + args.test_size,
        pass_fraction=args.pass_fraction,
        seed=args.seed,
    )
    train, test = split_chosen(
        chosen,
        pool,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_jsonl(args.train_out, train)
    write_jsonl(args.test_out, test)
    print(
        f"wrote {len(train)} train -> {args.train_out}; "
        f"{len(test)} test -> {args.test_out}"
    )


if __name__ == "__main__":
    main()
