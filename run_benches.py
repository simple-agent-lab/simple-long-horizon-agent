#!/usr/bin/env python3
"""Run ClawEvalkit benches via simple-agent-lab framework.

Usage:
    python3 run_benches.py --bench clawbench_tribe --model <model> [--sample 3]
"""
from __future__ import annotations

import argparse
import json
import os
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
        "response_preview": clean[:500],
        "check_type": check_type,
        "expected": expected,
    }


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
    args = parser.parse_args()

    # Bypass proxy for internal API endpoints
    os.environ["NO_PROXY"] = "search.bytedance.net,ark-cn-beijing.bytedance.net,localhost,127.0.0.1"
    os.environ["no_proxy"] = "search.bytedance.net,ark-cn-beijing.bytedance.net,localhost,127.0.0.1"

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

    # Set up provider env for simple-agent-lab
    # The framework expects OPENAI_MODEL, OPENAI_AUTH_TOKEN, OPENAI_BASE_URL
    # We'll use the Azure GPT endpoint
    # Azure GPT endpoint (OpenAI-compatible format)
    # Format: {azure_endpoint}/openai/deployments/{model}
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/"
        + args.model,
    )
    api_key = os.environ.get("GPT_API_KEY", "")
    if not api_key:
        print("ERROR: GPT_API_KEY not set. Load from 0001_utils/api/.env")
        sys.exit(1)

    provider_env = {
        "OPENAI_MODEL": args.model,
        "OPENAI_AUTH_TOKEN": api_key,
        "OPENAI_BASE_URL": base_url,
    }

    # Also set in os.environ so the LLM adapter can read them at runtime
    for k, v in provider_env.items():
        os.environ[k] = v

    # Import suite
    if args.bench == "clawbench_tribe":
        from evals.clawbench_tribe.suite import ClawBenchTribeSuite

        suite = ClawBenchTribeSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = score_tribe
    elif args.bench == "pinchbench":
        from evals.pinchbench.suite import PinchBenchSuite

        suite = PinchBenchSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None  # scoring is done in-container via evaluate hook
    elif args.bench == "clawbench_official":
        from evals.clawbench_official.suite import ClawBenchOfficialSuite

        suite = ClawBenchOfficialSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None
    elif args.bench == "skillsbench":
        from evals.skillsbench.suite import SkillsBenchSuite

        suite = SkillsBenchSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None
    elif args.bench == "zclawbench":
        from evals.zclawbench.suite import ZClawBenchSuite

        suite = ZClawBenchSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None  # host-side LLM judge needed
    elif args.bench == "agentbench":
        from evals.agentbench.suite import AgentBenchSuite

        suite = AgentBenchSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None
    elif args.bench == "claweval":
        from evals.claweval.suite import ClawEvalSuite

        suite = ClawEvalSuite()
        instances = suite.load_instances()[: args.sample]
        score_fn = None  # host-side LLM judge needed
    else:
        print(f"ERROR: Unknown bench '{args.bench}'")
        sys.exit(1)

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
            )

            # Host-side scoring
            if score_fn:
                score_result = score_fn(artifacts.run_dir, instance)
                score_result["instance_id"] = instance_id
                results.append(score_result)
                status = "PASSED" if score_result["passed"] else "FAILED"
                print(f"  -> {status} (score: {score_result['score']})")
            else:
                print(f"  -> completed (status: {artifacts.status_code})")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"instance_id": instance_id, "passed": False, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    avg_score = sum(r.get("score", 0) for r in results) / total if total else 0
    print(f"Bench: {args.bench}")
    print(f"Results: {passed}/{total} passed")
    print(f"Average score: {avg_score:.1f}")
    print(f"Run directory: {run_root}")


if __name__ == "__main__":
    main()
