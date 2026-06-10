#!/usr/bin/env python3
"""Run ClawEvalkit benches via simple-agent-lab framework.

Usage:
    python3 run_benches.py --bench clawbench_tribe --model <model> [--sample 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simple_agent_lab.evals import (
    LocalDirStore,
    LocalDockerBackend,
    LocalProcessBackend,
    run_suite_instance,
)
from simple_agent_lab.evals.protocols import LaunchSpec


def _extract_last_assistant_text(trajectory_path: Path) -> str:
    """从 trajectory.jsonl 中提取最后一条 assistant 消息的文本。"""
    if not trajectory_path.exists():
        return ""
    lines = trajectory_path.read_text().strip().split("\n")
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        # trace_record format: messages is a list of serialized Message dicts
        for msg in reversed(entry.get("messages", [])):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Content blocks: [{"kind": "text", "text": "..."}, ...]
                texts = []
                for block in content:
                    if isinstance(block, str):
                        texts.append(block)
                    elif isinstance(block, dict):
                        if block.get("kind") == "text" or block.get("type") == "text":
                            texts.append(block.get("text", ""))
                if texts:
                    return "\n".join(texts)
    return ""


def score_tribe(run_dir: Path, instance: dict) -> dict:
    """Host-side scoring for tribe: read trajectory, check expected answer."""
    trajectory_path = run_dir / "out" / "trajectory.jsonl"
    response = _extract_last_assistant_text(trajectory_path)

    if not response:
        return {"passed": False, "score": 0, "error": "no response in trajectory"}

    # Clean reasoning tags
    clean = re.sub(r"ILA.*?wiat", "", response, flags=re.DOTALL)
    clean = re.sub(r"<reasoning>.*?</reasoning>", "", clean, flags=re.DOTALL)

    check_type = instance.get("check_type", "contains")
    expected = instance.get("expected", "")

    passed = False
    if check_type == "contains":
        passed = expected in clean
    elif check_type == "json_check":
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                passed = isinstance(d, dict) and "name" in d
            except json.JSONDecodeError:
                pass
    elif check_type == "quality_check":
        passed = "REST" in clean and "GraphQL" in clean and len(clean) > 300

    return {
        "passed": passed,
        "score": 100 if passed else 0,
        "scoring_status": "scored",
        "score_source": "host_trajectory_check",
        "response_preview": clean[:500],
        "check_type": check_type,
        "expected": expected,
    }


def _load_result_json(run_dir: Path) -> dict:
    result_path = run_dir / "out" / "result.json"
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"invalid result.json: {exc}"}


def _coerce_score(value: object) -> float | None:
    if isinstance(value, bool):
        return 100.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_result(
    run_dir: Path,
    instance: dict,
    *,
    bench: str,
    pass_threshold: float,
) -> dict:
    """Summarize the suite result artifact for CLI reporting.

    This deliberately distinguishes real benchmark scores from "completed but
    judge missing" cases, so zclawbench/claweval do not look better than they
    are until their host-side judges are wired.
    """
    instance_id = instance.get("instance_id", run_dir.name)
    result = _load_result_json(run_dir)
    summary = {
        "instance_id": instance_id,
        "passed": False,
        "score": 0.0,
        "scoring_status": "scoring_pending",
        "score_source": "none",
    }
    if not result:
        summary["error"] = "missing result.json"
        return summary
    if "error" in result and "invalid result.json" in str(result["error"]):
        summary["error"] = result["error"]
        return summary

    score_source = None
    score = None
    for key in ("score", "layer0_score"):
        score = _coerce_score(result.get(key))
        if score is not None:
            score_source = key
            break
    if score is None and isinstance(result.get("mean"), (int, float)):
        score = float(result["mean"]) * 100
        score_source = "mean"

    if score is None:
        summary.update({
            "status": result.get("status", "completed"),
            "note": "result artifact exists, but no benchmark judge score is available yet",
        })
        if bench in {"zclawbench", "claweval"}:
            summary["score_source"] = "missing_host_judge"
            summary["note"] = "host-side LLM/user-agent judge is not implemented yet"
        return summary

    passed = bool(result["passed"]) if "passed" in result else score >= pass_threshold
    summary.update({
        "passed": passed,
        "score": round(score, 4),
        "scoring_status": "scored",
        "score_source": score_source or "result",
        "pass_threshold": pass_threshold,
    })
    for key in (
        "status",
        "tests_passed",
        "tests_failed",
        "tests_errors",
        "tests_total",
        "earned_points",
        "total_points",
    ):
        if key in result:
            summary[key] = result[key]
    if bench in {"zclawbench", "claweval"}:
        summary["scoring_status"] = "scoring_pending"
        summary["score_source"] = "missing_host_judge"
        summary["note"] = "host-side LLM/user-agent judge is not implemented yet"
    return summary


def select_instances(
    instances: list[dict],
    *,
    sample: int,
    strategy: str,
    seed: int,
) -> list[dict]:
    """Select a benchmark subset.

    The default "head" mode preserves the previous behavior. "spread" samples
    evenly across the loaded instance order, which is useful for ClawBench-style
    directories where sorted order clusters tasks by domain. "random" is
    deterministic under --seed.
    """
    if sample <= 0 or sample >= len(instances):
        return list(instances)
    if strategy == "head":
        return instances[:sample]
    if strategy == "random":
        rng = random.Random(seed)
        picks = rng.sample(range(len(instances)), sample)
        return [instances[i] for i in sorted(picks)]
    if strategy == "spread":
        if sample == 1:
            return [instances[0]]
        last = len(instances) - 1
        indexes = [round(i * last / (sample - 1)) for i in range(sample)]
        # round() can collide for small datasets; preserve order and backfill.
        seen: set[int] = set()
        ordered: list[int] = []
        for idx in indexes:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        for idx in range(len(instances)):
            if len(ordered) >= sample:
                break
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return [instances[i] for i in sorted(ordered[:sample])]
    raise ValueError(f"unknown sample strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description="Run ClawEvalkit benches")
    parser.add_argument("--bench", required=True, help="Benchmark name")
    parser.add_argument("--model", default="gpt-4o-2024-11-20", help="Model name for API")
    parser.add_argument("--sample", type=int, default=3, help="Number of tasks to run")
    parser.add_argument(
        "--backend",
        choices=["process", "docker"],
        default="process",
        help="Execution backend",
    )
    parser.add_argument("--max-turns", type=int, default=10, help="Max agent turns")
    parser.add_argument("--run-root", default="runs", help="Output directory")
    parser.add_argument(
        "--api-kind",
        choices=["openai-chat", "openai-responses"],
        default=os.environ.get("API_KIND", "openai-chat"),
        help="Provider API wire protocol.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override OPENAI_BASE_URL for this run.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=os.environ.get("OPENAI_REASONING_EFFORT", ""),
        help="Responses API reasoning effort, e.g. high for GPT-5.x thinking models.",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("OPENAI_SESSION_ID", ""),
        help="Optional ModelHub session id header value.",
    )
    parser.add_argument(
        "--log-id",
        default=os.environ.get("OPENAI_LOG_ID", ""),
        help="Optional ModelHub log id header value.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=100.0,
        help=(
            "Score threshold for pass/fail when result.json has no explicit "
            "'passed' field. Average score is unaffected."
        ),
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["head", "spread", "random"],
        default="head",
        help="Subset selection strategy; spread avoids domain-clustered first-N samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random sampling")
    args = parser.parse_args()

    # Bypass proxy for internal API endpoints
    no_proxy = (
        "search.bytedance.net,ark-cn-beijing.bytedance.net,aidp.bytedance.net,"
        "localhost,127.0.0.1"
    )
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy

    # Load API config. Repo-local .env is useful for ModelHub experiments and is
    # gitignored; 0001_utils remains the legacy fallback used by earlier runs.
    repo_env_path = PROJECT_ROOT / ".env"
    if repo_env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(repo_env_path, override=False)

    # Load API config from 0001_utils (symlink in repo root)
    utils_dir = PROJECT_ROOT / "0001_utils"
    api_env_path = utils_dir / "api" / ".env"
    if not api_env_path.exists():
        # Fallback: try parent structure
        utils_dir = PROJECT_ROOT.parent.parent / "0001_utils"
        api_env_path = utils_dir / "api" / ".env"
    if api_env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(api_env_path)

    # Set up provider env for simple-agent-lab. Chat-compatible runs keep the
    # legacy gateway default; Responses API runs use ModelHub by default.
    default_base_url = (
        "https://aidp.bytedance.net/api/modelhub/online/responses/openai/responses"
        if args.api_kind == "openai-responses"
        else "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/"
        + args.model
    )
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", default_base_url)
    api_key = os.environ.get("OPENAI_AUTH_TOKEN") or os.environ.get("GPT_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_AUTH_TOKEN or GPT_API_KEY not set.")
        sys.exit(1)

    provider_env = {
        "OPENAI_MODEL": args.model,
        "OPENAI_AUTH_TOKEN": api_key,
        "OPENAI_BASE_URL": base_url,
        "API_KIND": args.api_kind,
    }
    if args.reasoning_effort:
        provider_env["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    if args.session_id:
        provider_env["OPENAI_SESSION_ID"] = args.session_id
    if args.log_id:
        provider_env["OPENAI_LOG_ID"] = args.log_id

    # Also set in os.environ so the LLM adapter can read them at runtime
    for k, v in provider_env.items():
        os.environ[k] = v

    # Import suite
    if args.bench == "clawbench_tribe":
        from evals.clawbench_tribe.suite import ClawBenchTribeSuite

        suite = ClawBenchTribeSuite()
        all_instances = suite.load_instances()
        score_fn = score_tribe
    elif args.bench == "pinchbench":
        from evals.pinchbench.suite import PinchBenchSuite

        suite = PinchBenchSuite()
        all_instances = suite.load_instances()
        score_fn = None  # scoring is done in-container via evaluate hook
    elif args.bench == "clawbench_official":
        from evals.clawbench_official.suite import ClawBenchOfficialSuite

        suite = ClawBenchOfficialSuite()
        all_instances = suite.load_instances()
        score_fn = None
    elif args.bench == "skillsbench":
        from evals.skillsbench.suite import SkillsBenchSuite

        suite = SkillsBenchSuite()
        all_instances = suite.load_instances()
        score_fn = None
    elif args.bench == "zclawbench":
        from evals.zclawbench.suite import ZClawBenchSuite

        suite = ZClawBenchSuite()
        all_instances = suite.load_instances()
        score_fn = None  # host-side LLM judge needed
    elif args.bench == "agentbench":
        from evals.agentbench.suite import AgentBenchSuite

        suite = AgentBenchSuite()
        all_instances = suite.load_instances()
        score_fn = None
    elif args.bench == "claweval":
        from evals.claweval.suite import ClawEvalSuite

        suite = ClawEvalSuite()
        all_instances = suite.load_instances()
        score_fn = None  # host-side LLM judge needed
    else:
        print(f"ERROR: Unknown bench '{args.bench}'")
        sys.exit(1)
    instances = select_instances(
        all_instances,
        sample=args.sample,
        strategy=args.sample_strategy,
        seed=args.seed,
    )
    if len(all_instances) and len(instances) < args.sample:
        print(
            f"NOTE: requested {args.sample} tasks, but {args.bench} only has "
            f"{len(all_instances)} instances; running {len(instances)}."
        )
    print(
        f"Selected {len(instances)}/{len(all_instances)} instances "
        f"(strategy={args.sample_strategy}, seed={args.seed})."
    )

    # Set up backend and store
    run_root = PROJECT_ROOT / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    if args.backend == "docker":
        backend = LocalDockerBackend()
    else:
        backend = LocalProcessBackend()

    store = LocalDirStore(run_root)

    # Run instances
    results = []
    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        print(f"\n[{i+1}/{len(instances)}] Running {instance_id}...")

        try:
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=f"{args.bench}_{args.model}",
                provider="openai",
                max_turns=args.max_turns,
                provider_env=provider_env,
                api_kind=args.api_kind,
            )
            if artifacts.status_code != 0:
                results.append({
                    "instance_id": instance_id,
                    "passed": False,
                    "score": 0.0,
                    "scoring_status": "error",
                    "error": artifacts.logs,
                })
                print(f"  -> ERROR: {artifacts.logs}")
                continue

            # Host-side scoring
            if score_fn:
                score_result = score_fn(artifacts.run_dir, instance)
                score_result["instance_id"] = instance_id
                results.append(score_result)
                status = "PASSED" if score_result["passed"] else "FAILED"
                print(f"  -> {status} (score: {score_result['score']})")
            else:
                score_result = summarize_result(
                    artifacts.run_dir,
                    instance,
                    bench=args.bench,
                    pass_threshold=args.pass_threshold,
                )
                results.append(score_result)
                if score_result.get("scoring_status") == "scored":
                    status = "PASSED" if score_result["passed"] else "FAILED"
                    print(
                        f"  -> {status} "
                        f"(score: {score_result['score']}, "
                        f"source: {score_result['score_source']})"
                    )
                else:
                    print(
                        "  -> completed "
                        f"(status: {artifacts.status_code}, "
                        f"scoring: {score_result['scoring_status']})"
                    )

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                "instance_id": instance_id,
                "passed": False,
                "score": 0.0,
                "scoring_status": "error",
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    avg_score = sum(r.get("score", 0) for r in results) / total if total else 0
    print(f"Bench: {args.bench}")
    print(f"Results: {passed}/{total} passed")
    print(f"Average score: {avg_score:.1f}")
    print(f"Run directory: {run_root}")
    summary_path = run_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "bench": args.bench,
                "model": args.model,
                "api_kind": args.api_kind,
                "reasoning_effort": args.reasoning_effort or None,
                "backend": args.backend,
                "sample": args.sample,
                "sample_strategy": args.sample_strategy,
                "seed": args.seed,
                "pass_threshold": args.pass_threshold,
                "available_instances": len(all_instances),
                "selected_instances": [i.get("instance_id") for i in instances],
                "passed": passed,
                "total": total,
                "average_score": round(avg_score, 4),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
