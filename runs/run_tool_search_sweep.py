"""Run a tool-search execution sweep across tool-universe sizes.

The single-run script is good for smoke checks. This script is for experimental
comparisons: it sweeps distractor counts and seeds, runs one or more exposure
modes, and writes both JSON and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.tool_search.synthetic import (  # noqa: E402
    BenchMode,
    BenchReport,
    BenchRunner,
    build_synthetic_registry,
    default_tasks,
    run_bench,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    load_dotenv,
    provider_from_env,
    request_extra_from_env,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distractors",
        default="0,25,100,250",
        help="Comma-separated distractor counts.",
    )
    parser.add_argument(
        "--seeds",
        default="7",
        help="Comma-separated synthetic registry seeds.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--static-tool-limit", type=int, default=64)
    parser.add_argument("--runner", choices=["scripted", "llm"], default="scripted")
    parser.add_argument(
        "--modes",
        default="proxy,dynamic_topk,static_budgeted",
        help="Comma-separated modes: proxy,dynamic_topk,static_budgeted.",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=0,
        help="Limit tasks per condition; useful for LLM budgeted sweeps.",
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--run-id",
        default=f"tool-search-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--out-root", default="evals/out/tool_search")
    return parser


def main() -> None:
    args = _parser().parse_args()
    distractors = _int_list(args.distractors)
    seeds = _int_list(args.seeds)
    modes = _modes(args.modes)
    runner: BenchRunner = args.runner

    provider = None
    request_extra = None
    if runner == "llm":
        load_dotenv(args.dotenv)
        provider = provider_from_env(read_reasoning=True, label="tool-search sweep")
        request_extra = _request_extra_for_tool_search()

    tasks = default_tasks()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Tool Search Sweep ===")
    print(f"runner: {runner}")
    print(f"distractors: {distractors}")
    print(f"seeds: {seeds}")
    print(f"modes: {modes}")
    print("")

    for distractor_count in distractors:
        for seed in seeds:
            registry = build_synthetic_registry(
                distractors=distractor_count,
                seed=seed,
            )
            for mode in modes:
                report = run_bench(
                    registry=registry,
                    tasks=tasks,
                    mode=mode,
                    runner=runner,
                    top_k=args.top_k,
                    static_tool_limit=args.static_tool_limit,
                    provider=provider,
                    request_extra=request_extra,
                    max_turns=args.max_turns,
                )
                summary = {
                    **report.summary(),
                    "distractors": distractor_count,
                    "total_tools": len(registry.records),
                    "seed": seed,
                    "top_k": args.top_k,
                    "static_tool_limit": args.static_tool_limit,
                }
                rows.append(summary)
                reports.append(_report_payload(report, summary))
                task_rows.extend(_task_rows(report, summary))
                print(
                    f"d={distractor_count:<5} seed={seed:<3} {mode:<16} "
                    f"success={summary['success_rate']:.3f} "
                    f"gold_hit={summary['gold_in_candidates_rate']:.3f} "
                    f"schema={summary['mean_schema_tokens']:.1f} "
                    f"peak={summary['mean_peak_context_tokens']:.1f}"
                )

    json_path = out_dir / "sweep.json"
    csv_path = out_dir / "summary.csv"
    task_csv_path = out_dir / "tasks.csv"
    json_path.write_text(
        json.dumps({"summaries": rows, "reports": reports}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_csv(task_csv_path, task_rows)
    print("")
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {task_csv_path}")


def _report_payload(report: BenchReport, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": summary,
        "results": [result.as_dict() for result in report.results],
    }


def _task_rows(report: BenchReport, summary: dict[str, Any]) -> list[dict[str, Any]]:
    condition = {
        "mode": summary["mode"],
        "runner": summary["runner"],
        "distractors": summary["distractors"],
        "total_tools": summary["total_tools"],
        "seed": summary["seed"],
        "top_k": summary["top_k"],
        "static_tool_limit": summary["static_tool_limit"],
    }
    return [{**condition, **result.as_dict()} for result in report.results]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise SystemExit("expected at least one integer")
    if any(item < 0 for item in items):
        raise SystemExit("integer lists must be non-negative")
    return items


def _modes(value: str) -> tuple[BenchMode, ...]:
    allowed = {"proxy", "dynamic_topk", "static_budgeted"}
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}")
    return items  # type: ignore[return-value]


def _request_extra_for_tool_search() -> dict[str, Any]:
    extra = request_extra_from_env()
    jwt = os.environ.get("CF_AIG_SIGNED_JWT", "").strip()
    if not jwt:
        return extra
    headers = dict(extra.get("extra_headers") or {})
    headers.setdefault("cf-aig-authorization", f"Bearer {jwt}")
    headers.setdefault("User-Agent", "simple-agent-lab/tool-search-sweep")
    return {**extra, "extra_headers": headers}


if __name__ == "__main__":
    main()
