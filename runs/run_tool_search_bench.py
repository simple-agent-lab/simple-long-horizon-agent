"""Run the local tool-search execution benchmark.

This is the first experimental harness for search+execute. By default it uses a
deterministic scripted runner (no model credentials, no Docker). Pass
``--runner llm`` to run the same tool universes with an OpenAI-compatible model.
It compares:

- proxy: expose only search_tools + invoke_tool;
- dynamic_topk: retrieve top-k, then expose those real tools;
- static_budgeted: expose the first N tools, simulating a static schema budget.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.tool_search.synthetic import (  # noqa: E402
    BenchMode,
    BenchRunner,
    build_synthetic_registry,
    run_bench,
    write_report,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    load_dotenv,
    provider_from_env,
    request_extra_from_env,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distractors", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--static-tool-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runner", choices=["scripted", "llm"], default="scripted")
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--mode",
        choices=["proxy", "dynamic_topk", "static_budgeted", "all"],
        default="all",
    )
    parser.add_argument(
        "--run-id",
        default=f"tool-search-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--out-root", default="evals/out/tool_search")
    return parser


def main() -> None:
    args = _parser().parse_args()
    registry = build_synthetic_registry(
        distractors=args.distractors,
        seed=args.seed,
    )
    modes: tuple[BenchMode, ...] = (
        ("proxy", "dynamic_topk", "static_budgeted")
        if args.mode == "all"
        else (args.mode,)
    )
    runner: BenchRunner = args.runner
    provider = None
    request_extra = None
    if runner == "llm":
        load_dotenv(args.dotenv)
        provider = provider_from_env(read_reasoning=True, label="tool-search bench")
        request_extra = _request_extra_for_tool_search()
    reports = [
        run_bench(
            registry=registry,
            mode=mode,
            runner=runner,
            top_k=args.top_k,
            static_tool_limit=args.static_tool_limit,
            provider=provider,
            request_extra=request_extra,
            max_turns=args.max_turns,
        )
        for mode in modes
    ]
    out = Path(args.out_root) / args.run_id / "results.json"
    write_report(out, reports)

    print("=== Tool Search Execution Bench ===")
    print(f"runner: {runner}")
    print(f"tools: {len(registry.records)}  distractors: {args.distractors}")
    print(f"top-k: {args.top_k}  static-tool-limit: {args.static_tool_limit}")
    print("")
    for report in reports:
        summary = report.summary()
        print(
            f"{summary['mode']:16} "
            f"success={summary['success_rate']:.3f} "
            f"correct_tool={summary['correct_tool_rate']:.3f} "
            f"schema_tokens={summary['mean_schema_tokens']:.1f} "
            f"peak_context={summary['mean_peak_context_tokens']:.1f} "
            f"errors={summary['errors']}"
        )
    print("")
    print(f"wrote: {out}")


def _request_extra_for_tool_search() -> dict:
    """Request extras for the local bench, including the CF AIG gateway header."""

    extra = request_extra_from_env()
    jwt = os.environ.get("CF_AIG_SIGNED_JWT", "").strip()
    if not jwt:
        return extra
    headers = dict(extra.get("extra_headers") or {})
    headers.setdefault("cf-aig-authorization", f"Bearer {jwt}")
    headers.setdefault("User-Agent", "simple-agent-lab/tool-search-bench")
    return {**extra, "extra_headers": headers}


if __name__ == "__main__":
    main()
