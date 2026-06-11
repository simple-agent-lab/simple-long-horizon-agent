"""Run OneMillion-Bench cases through the generic `Suite` framework (ADR 0017).

The OneMillion-Bench run entry, mirroring ``runs/run_swebench_suite.py`` but for
a light, Docker-free suite: it drives the container half through
``run_suite_instance(OneMillionSuite, LocalProcessBackend, LocalDirStore)`` — the
same primitive every suite uses. Generation is one tool-free model turn; the
in-environment ``evaluate`` hook grades the answer against the case's rubrics
with a judge model.

Usage (a downloaded dataset under ``datasets/OneMillion-Bench/``):

    # one case by id
    uv run python runs/run_onemillion_suite.py case_2860 \
        --dataset datasets/OneMillion-Bench/healthcare_and_medicine

    # a whole domain (or the full dataset)
    uv run python runs/run_onemillion_suite.py --all \
        --dataset datasets/OneMillion-Bench --concurrency 8

Reads the generator OPENAI_MODEL / OPENAI_AUTH_TOKEN (+ optional OPENAI_BASE_URL)
and the judge JUDGE_MODEL / JUDGE_AUTH_TOKEN (each falling back to the OPENAI_*
value) from ``.env``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.onemillion import harness  # noqa: E402
from evals.onemillion.suite import OneMillionSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalProcessBackend,
    run_dataset,
    run_suite_instance,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instance_id",
        nargs="?",
        default=None,
        help="Case id (e.g. case_2860). Omit with --all to run the whole dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=str(harness.DEFAULT_DATASET_DIR),
        help="Dataset file or directory (default: datasets/OneMillion-Bench).",
    )
    parser.add_argument("--all", action="store_true", help="Run every case found.")
    parser.add_argument("--limit", type=int, default=0, help="Cap cases in --all mode.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument(
        "--run-id", default=f"onemillion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument("--api-kind", default=None)
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--run-root", default=str(harness.DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--no-scoring",
        action="store_true",
        help="Capture answers only; skip the in-environment rubric judge.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Load .env into the process so both the generator (provider_env) and the
    # judge (read from os.environ by the in-process evaluate hook) are populated.
    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness.container_environment(args.provider)
    api_kind = harness.resolve_api_kind(args.api_kind)
    provider_env[harness.API_KIND_ENV] = api_kind

    suite = OneMillionSuite(in_env_scoring=not args.no_scoring)
    backend = LocalProcessBackend()
    store = LocalDirStore(Path(args.run_root))
    run_root = Path(args.run_root)

    common = dict(
        suite=suite,
        backend=backend,
        store=store,
        run_root=run_root,
        run_id=args.run_id,
        provider=args.provider,
        api_kind=api_kind,
        max_turns=args.max_turns,
        provider_env=provider_env,
    )

    if args.all:
        instances = harness.load_dataset(args.dataset)
        if args.limit > 0:
            instances = instances[: args.limit]
        print(f"==> Running {len(instances)} OneMillion-Bench cases")
        print(f"    run-id: {args.run_id}  concurrency: {args.concurrency}")
        report = run_dataset(
            instances=instances,
            concurrency=args.concurrency,
            on_result=lambda r: print(
                f"    {r.instance_id}: {'ok' if r.ok else r.error}"
            ),
            **common,
        )
        summary = report.summary()
        print(f"==> done: {summary}")
        print(f"    artifacts under: {run_root / args.run_id}")
        if summary.get("failed"):
            raise SystemExit(1)
        return

    instance = harness.load_case(args.dataset, args.instance_id)
    instance_id = str(instance["instance_id"])
    print("==> Running OneMillion-Bench case through OneMillionSuite")
    print(f"    case:      {instance_id}")
    print(f"    run-id:    {args.run_id}")
    print(f"    scoring:   {'on' if not args.no_scoring else 'off'}")
    print("")

    result = run_suite_instance(instance=instance, **common)
    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print("")
    print(f"==> run dir: {result.run_dir}")
    print(f"    result:  {result.run_dir / 'out' / 'result.json'}")
    print(f"    status:  {result.status_code}")
    if result.status_code != 0:
        raise SystemExit(result.status_code)


if __name__ == "__main__":
    main()
