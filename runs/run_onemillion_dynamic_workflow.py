"""Run OneMillion-Bench cases with a Dynamic Workflow JavaScript runtime.

The run uses a facade agent: it generates or loads ``workflow.js``, executes it
with the dynamic workflow runtime, and writes the final answer to the normal
OneMillion result seam.

Usage:

    uv run python runs/run_onemillion_dynamic_workflow.py case_10086 \
        --dataset datasets/OneMillion-Bench/healthcare_and_medicine \
        --no-scoring

    # Deterministic smoke without credentials:
    uv run python runs/run_onemillion_dynamic_workflow.py \
        --dataset /path/to/tiny.json --provider fake --no-scoring
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
from evals.onemillion.suite import OneMillionDynamicWorkflowSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalProcessBackend,
    run_dataset,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.onemillion.dynamic_workflow_container import (  # noqa: E402
    DYNAMIC_WORKFLOW_MAX_AGENTS_ENV,
    DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV,
    DYNAMIC_WORKFLOW_SCRIPT_ENV,
    DYNAMIC_WORKFLOW_TIMEOUT_ENV,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", nargs="?", default=None)
    parser.add_argument("--dataset", default=str(harness.DEFAULT_DATASET_DIR))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument(
        "--run-id",
        default=f"onemillion-dynwf-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument("--api-kind", default=None)
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--run-root", default=str(harness.DEFAULT_RUN_ROOT))
    parser.add_argument("--workflow-script", default="")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-agents", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-scoring", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness.container_environment(args.provider)
    api_kind = harness.resolve_api_kind(args.api_kind)
    provider_env[harness.API_KIND_ENV] = api_kind

    workflow_env = {
        DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV: str(args.max_concurrency),
        DYNAMIC_WORKFLOW_MAX_AGENTS_ENV: str(args.max_agents),
        DYNAMIC_WORKFLOW_TIMEOUT_ENV: str(args.timeout),
    }
    if args.workflow_script:
        workflow_env[DYNAMIC_WORKFLOW_SCRIPT_ENV] = str(Path(args.workflow_script))
    os.environ.update(workflow_env)
    provider_env.update(workflow_env)

    suite = OneMillionDynamicWorkflowSuite(in_env_scoring=not args.no_scoring)
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
    print("==> Running OneMillion-Bench case through Dynamic Workflow")
    print(f"    case:      {instance_id}")
    print(f"    run-id:    {args.run_id}")
    print(f"    scoring:   {'on' if not args.no_scoring else 'off'}")
    result = run_suite_instance(instance=instance, **common)
    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print("")
    print(f"==> run dir: {result.run_dir}")
    print(f"    result:  {result.run_dir / 'out' / 'result.json'}")
    print("    dynamic workflow details are under result['dynamic_workflow']")
    print(f"    status:  {result.status_code}")
    if result.status_code != 0:
        raise SystemExit(result.status_code)


if __name__ == "__main__":
    main()
