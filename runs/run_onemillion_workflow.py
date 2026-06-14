"""Run OneMillion-Bench cases answered by a multi-agent *workflow*.

A sibling of ``runs/run_onemillion_suite.py`` that swaps the single tool-free
generator for one of the ``simple_agent_lab.workflow`` orchestrations. Same
host plumbing (dataset loading, rubric staging, judge scoring); the only
differences are the suite (``OneMillionWorkflowSuite``) and a ``--workflow``
flag that selects which orchestration generates the answer.

Usage (a downloaded dataset under ``datasets/OneMillion-Bench/``):

    # one case with the reflection workflow
    uv run python runs/run_onemillion_workflow.py case_10086 \
        --workflow reflection \
        --dataset datasets/OneMillion-Bench/healthcare_and_medicine

    # a whole domain with planner/executor
    uv run python runs/run_onemillion_workflow.py --all --workflow planner_executor \
        --dataset datasets/OneMillion-Bench/law --concurrency 8

Workflows: single | reflection | planner_executor | parallel | chain | routing.
``--reflection-rounds`` and ``--parallel-workers`` tune those two. Reads the
generator OPENAI_* and judge JUDGE_* env (each judge var falling back to the
matching OPENAI_*) from ``.env``.
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

from evals.onemillion import harness  # noqa: E402
from evals.onemillion.suite import OneMillionWorkflowSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalProcessBackend,
    run_dataset,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.onemillion.workflow_container import (  # noqa: E402
    PARALLEL_WORKERS_ENV,
    REFLECTION_ROUNDS_ENV,
    TIMEOUT_ENV,
    WORKFLOW_CHOICES,
    WORKFLOW_ENV,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instance_id",
        nargs="?",
        default=None,
        help="Case id (e.g. case_10086). Omit with --all to run the whole dataset.",
    )
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_CHOICES),
        default="reflection",
        help="Which workflow generates the answer (default: reflection).",
    )
    parser.add_argument(
        "--reflection-rounds",
        type=int,
        default=None,
        help="Critique/revise rounds for --workflow reflection.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Worker count for --workflow parallel.",
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
    # The facade returns on its first turn; the workflow's own sub-agents carry
    # their own turn budgets, so the outer loop needs only one turn.
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument(
        "--run-id",
        default=f"onemillion-wf-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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

    # The container half (in-process for LocalProcessBackend) reads these from
    # os.environ; mirror them into provider_env so containerized backends get
    # them too. Set them before the suite runs.
    workflow_env = {WORKFLOW_ENV: args.workflow}
    if args.reflection_rounds is not None:
        workflow_env[REFLECTION_ROUNDS_ENV] = str(args.reflection_rounds)
    if args.parallel_workers is not None:
        workflow_env[PARALLEL_WORKERS_ENV] = str(args.parallel_workers)
    if args.timeout is not None:
        workflow_env[TIMEOUT_ENV] = str(args.timeout)
    os.environ.update(workflow_env)
    provider_env.update(workflow_env)

    suite = OneMillionWorkflowSuite(in_env_scoring=not args.no_scoring)
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
        print(f"==> Running {len(instances)} OneMillion-Bench cases [{args.workflow}]")
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
    print("==> Running OneMillion-Bench case through OneMillionWorkflowSuite")
    print(f"    case:      {instance_id}")
    print(f"    workflow:  {args.workflow}")
    print(f"    run-id:    {args.run_id}")
    print(f"    scoring:   {'on' if not args.no_scoring else 'off'}")
    print("")

    result = run_suite_instance(instance=instance, **common)
    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print("")
    print(f"==> run dir: {result.run_dir}")
    print(f"    result:  {result.run_dir / 'out' / 'result.json'}")
    print("    steps:   per-step breakdown is under the 'workflow' key in result.json")
    print(f"    status:  {result.status_code}")
    if result.status_code != 0:
        raise SystemExit(result.status_code)


if __name__ == "__main__":
    main()
