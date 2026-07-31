"""Run OneMillion-Bench cases through the generic `Suite` framework.

The OneMillion-Bench run entry, mirroring ``runs/_benches/swebench.py`` but for
a light, Docker-free suite: it drives the container half through
``run_suite_instance(OneMillionSuite, LocalProcessBackend, LocalDirStore)`` — the
same primitive every suite uses. The in-environment ``evaluate`` hook grades the
answer against the case's rubrics with a judge model.

Generation strategy is chosen with the shared ``--agent-flavor`` knob (like
SWE-bench): ``single`` (default) is one tool-free model turn; a workflow flavor
(``reflection`` / ``planner_executor`` / ``parallel`` / ``chain`` / ``routing``
/ ``pdr``) produces the answer with a multi-agent ``simple_long_horizon_agent.workflow``
orchestration. There is one OneMillion entry — the flavor picks the strategy.

Usage (a downloaded dataset under ``datasets/OneMillion-Bench/``):

    # one case, single tool-free turn (default)
    uv run python -m runs.run_bench onemillion case_2860 \
        --dataset datasets/OneMillion-Bench/healthcare_and_medicine

    # a whole domain via the reflection workflow
    uv run python -m runs.run_bench onemillion --all --agent-flavor reflection \
        --dataset datasets/OneMillion-Bench --concurrency 8

Reads the generator OPENAI_MODEL / OPENAI_AUTH_TOKEN (+ optional OPENAI_BASE_URL)
and the judge JUDGE_MODEL / JUDGE_AUTH_TOKEN (each falling back to the OPENAI_*
value) from ``.env``.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import simple_long_horizon_agent.config as config
from evals.onemillion import harness
from evals.onemillion.suite import OneMillionSuite
from simple_long_horizon_agent.agent_flavors import AGENT_FLAVOR_ENV
from simple_long_horizon_agent.evals import (
    LocalDirStore,
    LocalProcessBackend,
    parse_with_profile,
    run_dataset,
    run_suite_instance,
)
from simple_long_horizon_agent.evals.suites.onemillion.container import OMB_FLAVORS

ROOT = Path(__file__).resolve().parents[2]

# Identity for the unified entry (runs/run_bench.py). `run(args)` returns a
# result dict so the dispatcher / dashboard can read a machine-readable outcome.
NAME = "onemillion"
DESCRIPTION = (
    "OneMillion-Bench: rubric-judged Q&A; --agent-flavor picks single (tool-free "
    "turn) or a multi-agent workflow (in-process; supports --all sweeps)."
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
        "--profile",
        default=None,
        help=(
            "Path to a JSON run-profile (its `env` fills env gaps, its `run` "
            "flags are defaults overridable by explicit flags)."
        ),
    )
    parser.add_argument(
        "--agent-flavor",
        choices=list(OMB_FLAVORS),
        default="single",
        help=(
            "Generation strategy (default: single). A workflow flavor "
            "(reflection|planner_executor|parallel|chain|routing|pdr) answers via "
            "a multi-agent simple_long_horizon_agent.workflow orchestration."
        ),
    )
    parser.add_argument(
        "--reflection-rounds",
        type=int,
        default=None,
        help="Critique/revise rounds for --agent-flavor reflection.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Worker count for --agent-flavor parallel.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout (s) for each sub-agent (default: 600).",
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


def run(args: argparse.Namespace) -> dict:
    # Load .env into the process so both the generator (provider_env) and the
    # judge (read from os.environ by the in-process evaluate hook) are populated.
    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness.container_environment(args.provider)
    api_kind = harness.resolve_api_kind(args.api_kind)
    provider_env[harness.API_KIND_ENV] = api_kind

    # The container half (in-process for LocalProcessBackend) reads AGENT_FLAVOR
    # (single vs a workflow) and the OMB_* workflow knobs from os.environ; mirror
    # them into provider_env so containerized backends get them too.
    flavor_env = {AGENT_FLAVOR_ENV: args.agent_flavor}
    if args.reflection_rounds is not None:
        flavor_env[config.OMB_REFLECTION_ROUNDS.name] = str(args.reflection_rounds)
    if args.parallel_workers is not None:
        flavor_env[config.OMB_PARALLEL_WORKERS.name] = str(args.parallel_workers)
    if args.timeout is not None:
        flavor_env[config.OMB_TIMEOUT.name] = str(args.timeout)
    os.environ.update(flavor_env)
    provider_env.update(flavor_env)

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
        print(
            f"==> Running {len(instances)} OneMillion-Bench cases [{args.agent_flavor}]"
        )
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
        return {
            "bench": NAME,
            "status_code": 1 if summary.get("failed") else 0,
            "run_dir": str(run_root / args.run_id),
            "result_path": None,
            "summary": summary,
        }

    instance = harness.load_case(args.dataset, args.instance_id)
    instance_id = str(instance["instance_id"])
    print("==> Running OneMillion-Bench case through OneMillionSuite")
    print(f"    case:      {instance_id}")
    print(f"    flavor:    {args.agent_flavor}")
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
    return {
        "bench": NAME,
        "status_code": result.status_code,
        "run_dir": str(result.run_dir),
        "result_path": str(result.run_dir / "out" / "result.json"),
        "summary": None,
    }


def main() -> None:
    raise SystemExit(run(parse_with_profile(_build_parser()))["status_code"])


if __name__ == "__main__":
    main()
