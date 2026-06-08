#!/usr/bin/env python3
"""Post-hoc judge for OpenClaw benchmark runs without deterministic scorers.

This script reads an existing run_benches.py summary.json and fills
scoring_pending entries with a host-side LLM judge score. It is intentionally
separate from the runner so we can distinguish official deterministic scores
from after-the-fact rubric judgments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def load_env() -> None:
    os.environ["NO_PROXY"] = "search.bytedance.net,ark-cn-beijing.bytedance.net,localhost,127.0.0.1"
    os.environ["no_proxy"] = "search.bytedance.net,ark-cn-beijing.bytedance.net,localhost,127.0.0.1"
    utils_dir = PROJECT_ROOT / "0001_utils"
    api_env_path = utils_dir / "api" / ".env"
    if not api_env_path.exists():
        api_env_path = PROJECT_ROOT.parent.parent / "0001_utils" / "api" / ".env"
    if api_env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(api_env_path)


def openai_client(*, model: str):
    from openai import OpenAI

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/"
        + model,
    )
    api_key = os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_AUTH_TOKEN")
    if not api_key:
        raise SystemExit("GPT_API_KEY or OPENAI_AUTH_TOKEN is required for judging")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_suite_instances(bench: str) -> dict[str, dict[str, Any]]:
    if bench == "pinchbench":
        from evals.pinchbench.suite import PinchBenchSuite

        suite = PinchBenchSuite()
    elif bench == "claweval":
        from evals.claweval.suite import ClawEvalSuite

        suite = ClawEvalSuite()
    elif bench == "zclawbench":
        from evals.zclawbench.suite import ZClawBenchSuite

        suite = ZClawBenchSuite()
    else:
        raise SystemExit(f"{bench} does not need this post-hoc judge")
    return {str(i["instance_id"]): i for i in suite.load_instances()}


def load_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "out" / "result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_trajectory(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for msg in record.get("messages", []):
            role = msg.get("role", "")
            kind = msg.get("kind", "")
            content = content_to_text(msg.get("content", []))
            if content:
                chunks.append(f"[{role}/{kind}] {content}")
    text = "\n".join(chunks)
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n...[trajectory truncated]...\n" + text[-max_chars // 2 :]


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            kind = block.get("kind") or block.get("type")
            if kind == "text":
                parts.append(str(block.get("text", "")))
            elif kind == "tool_call":
                parts.append(
                    "[tool_call "
                    + str(block.get("name", ""))
                    + " "
                    + json.dumps(block.get("arguments", {}), ensure_ascii=False)[:1200]
                    + "]"
                )
            elif kind == "tool_result":
                parts.append("[tool_result] " + content_to_text(block.get("content", [])))
    return "\n".join(p for p in parts if p).strip()


def build_judge_prompt(
    *,
    bench: str,
    instance: dict[str, Any],
    result: dict[str, Any],
    trajectory: str,
) -> tuple[str, str, str]:
    instance_id = str(instance.get("instance_id", ""))
    if bench == "pinchbench":
        rubric = instance.get("llm_judge_rubric") or instance.get("grading_criteria") or ""
        prompt = instance.get("prompt", "")
        reference = instance.get("expected_behavior", "")
        source = "host_llm_judge_rubric"
    elif bench == "claweval":
        rubric = instance.get("judge_rubric", "")
        prompt = instance.get("prompt_text", "")
        reference = instance.get("reference_solution", "")
        components = instance.get("scoring_components", [])
        rubric = rubric + "\n\nScoring components:\n" + json.dumps(
            components, ensure_ascii=False, indent=2
        )
        source = "host_llm_judge_rubric"
    elif bench == "zclawbench":
        category = instance.get("category", "")
        prompt = (
            f"Task id: {instance_id}\n"
            f"Category: {category}\n"
            "The agent was asked to complete a category-appropriate task and save a concrete output."
        )
        rubric = (
            "Score whether the agent made a concrete, useful, category-appropriate "
            "attempt. Award 100 for a complete and specific deliverable, 50 for a "
            "partial but relevant deliverable, and 0 for asking for more information, "
            "refusing, or producing no substantive output."
        )
        reference = "No official reference is available in the current zclawbench adapter."
        source = "generic_host_llm_judge"
    else:
        raise AssertionError(bench)

    user = f"""Benchmark: {bench}
Instance: {instance_id}

Task prompt:
{prompt}

Reference / expected behavior:
{reference}

Rubric:
{rubric}

Result artifact:
{json.dumps(result, ensure_ascii=False, indent=2)[:4000]}

Compact trajectory:
{trajectory}

Return JSON only with this schema:
{{
  "score": <number from 0 to 100>,
  "passed": <true if score >= 60 else false>,
  "rationale": "<short explanation>",
  "failure_label_hint": "A|B|C|D|E|F|unknown"
}}
"""
    system = (
        "You are a strict benchmark judge. Use only the task, rubric, result "
        "artifact, and trajectory. Do not give credit for files that are merely "
        "claimed unless the trajectory or artifact supports the claim. Return valid JSON only."
    )
    return system, user, source


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("judge did not return a JSON object")
    return json.loads(match.group(0))


def judge_one(
    *,
    client: Any,
    model: str,
    bench: str,
    instance: dict[str, Any],
    run_dir: Path,
    max_trajectory_chars: int,
) -> dict[str, Any]:
    result = load_result(run_dir)
    trajectory = compact_trajectory(
        run_dir / "out" / "trajectory.jsonl", max_chars=max_trajectory_chars
    )
    system, user, source = build_judge_prompt(
        bench=bench, instance=instance, result=result, trajectory=trajectory
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        timeout=120,
    )
    content = response.choices[0].message.content or "{}"
    data = parse_json_object(content)
    score = max(0.0, min(100.0, float(data.get("score", 0))))
    return {
        "score": round(score, 2),
        "passed": bool(data.get("passed", score >= 60)),
        "scoring_status": "judged",
        "score_source": source,
        "judge_model": model,
        "rationale": str(data.get("rationale", ""))[:1000],
        "failure_label_hint": str(data.get("failure_label_hint", "unknown")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=["pinchbench", "zclawbench", "claweval"])
    parser.add_argument("--run-root", required=True, help="Directory containing summary.json")
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--only-pending", action="store_true", default=True)
    parser.add_argument("--max-trajectory-chars", type=int, default=12000)
    args = parser.parse_args()

    load_env()
    client = openai_client(model=args.model)
    run_root = PROJECT_ROOT / args.run_root
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    instances = load_suite_instances(args.bench)
    run_id = f"{args.bench}_{summary['model']}"

    updated = 0
    for item in summary["results"]:
        if args.only_pending and item.get("scoring_status") not in {"scoring_pending", "error"}:
            continue
        instance_id = str(item["instance_id"])
        instance = instances.get(instance_id)
        if not instance:
            item["judge_error"] = "instance not found"
            continue
        run_dir = run_root / run_id / instance_id
        try:
            judged = judge_one(
                client=client,
                model=args.model,
                bench=args.bench,
                instance=instance,
                run_dir=run_dir,
                max_trajectory_chars=args.max_trajectory_chars,
            )
        except Exception as exc:
            item["judge_error"] = str(exc)
            continue
        item.update(judged)
        updated += 1
        print(f"{args.bench}/{instance_id}: {judged['score']} {judged['score_source']}")

    scored = [r for r in summary["results"] if r.get("scoring_status") in {"scored", "judged"}]
    pending = [r for r in summary["results"] if r.get("scoring_status") == "scoring_pending"]
    summary["posthoc_judge_model"] = args.model
    summary["posthoc_judged"] = updated
    summary["passed"] = sum(1 for r in scored if r.get("passed"))
    summary["total"] = len(summary["results"])
    summary["average_score"] = round(
        sum(float(r.get("score", 0)) for r in scored) / len(scored), 4
    ) if scored else 0.0
    summary["scored_total"] = len(scored)
    summary["pending_total"] = len(pending)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {updated} entries in {summary_path}")


if __name__ == "__main__":
    main()
