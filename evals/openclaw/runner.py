"""Main evaluation runner for ClawEvalkit benchmarks via Simple Agent Lab.

Supports running SAL's Bash Agent against ClawEvalkit's PinchBench and
AgentBench benchmarks. Loads task definitions from ClawEvalkit's benchmark
directories, executes them with SAL, and produces summary results.

Usage:
    python -m evals.openclaw --bench pinchbench --model claude-sonnet --sample 3
    python -m evals.openclaw --bench agentbench --model glm-4.7 --clawevalkit /path/to/ClawEvalkit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adapter import run_pinchbench_task, run_agentbench_task, run_clawbench_task
from .config import ModelConfig, build_model_config


def _load_pinchbench_tasks(clawevalkit_dir: Path) -> list[dict]:
    """Load PinchBench tasks from ClawEvalkit's benchmarks/pinchbench/tasks/."""
    import re
    import yaml

    tasks_dir = clawevalkit_dir / "benchmarks" / "pinchbench" / "tasks"
    if not tasks_dir.exists():
        print(f"[ERROR] PinchBench tasks dir not found: {tasks_dir}", file=sys.stderr)
        return []

    def _extract_section(pattern: str, text: str) -> str:
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    tasks = []
    for md in sorted(tasks_dir.glob("*.md")):
        if md.name.startswith("TASK_TEMPLATE") or md.name.startswith("_"):
            continue
        content = md.read_text(encoding="utf-8")

        frontmatter = {}
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1))
            except Exception:
                pass

        prompt = ""
        prompt_match = re.search(r"## Prompt\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
        if prompt_match:
            prompt = prompt_match.group(1).strip()

        grade_code = ""
        checks_match = re.search(
            r"## Automated Checks.*?```python\s*\n(.*?)```", content, re.DOTALL
        )
        if checks_match:
            grade_code = checks_match.group(1).strip()

        tid = frontmatter.get("id", md.stem)
        timeout = int(frontmatter.get("timeout_seconds", 120))
        workspace_files = frontmatter.get("workspace_files", [])
        grading_type = frontmatter.get("grading_type", "automated")

        tasks.append(
            {
                "id": tid,
                "prompt": prompt,
                "grade_code": grade_code,
                "timeout": timeout,
                "workspace_files": workspace_files,
                "grading_type": grading_type,
            }
        )

    return tasks


def _load_agentbench_tasks(clawevalkit_dir: Path) -> list[dict]:
    """Load AgentBench tasks from ClawEvalkit's benchmarks/agentbench-openclaw/tasks/."""
    import yaml

    tasks_dir = clawevalkit_dir / "benchmarks" / "agentbench-openclaw" / "tasks"
    if not tasks_dir.exists():
        print(f"[ERROR] AgentBench tasks dir not found: {tasks_dir}", file=sys.stderr)
        return []

    tasks = []
    for cat_dir in sorted(tasks_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for task_dir in sorted(cat_dir.iterdir()):
            yaml_f = task_dir / "task.yaml"
            if yaml_f.exists():
                tasks.append(
                    {
                        "task_id": task_dir.name,
                        "category": cat_dir.name,
                        "yaml_path": str(yaml_f),
                    }
                )

    return tasks


def _find_clawbench_tasks_root(root: Path) -> Path | None:
    """Find ClawBench's native tasks directory across supported layouts."""
    candidates = [
        root / "tasks",
        root / "benchmarks" / "claw-bench" / "tasks",
        root / "assets" / "benchmarks" / "claw-bench" / "tasks",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_clawbench_tasks(clawevalkit_dir: Path) -> list[dict]:
    """Load ClawBench Official tasks from a native claw-bench tasks directory."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    tasks_root = _find_clawbench_tasks_root(clawevalkit_dir)
    if tasks_root is None:
        print(
            f"[ERROR] ClawBench tasks dir not found under: {clawevalkit_dir}",
            file=sys.stderr,
        )
        return []

    tasks = []
    for toml_path in sorted(tasks_root.glob("*/*/task.toml")):
        task_dir = toml_path.parent
        try:
            with open(toml_path, "rb") as f:
                raw = tomllib.load(f)
            if "task" in raw:
                raw = {**raw.pop("task"), **raw}
        except Exception as exc:
            print(f"[WARN] Failed to load {toml_path}: {exc}", file=sys.stderr)
            continue

        tasks.append(
            {
                "id": raw.get("id", task_dir.name),
                "dir_name": task_dir.name,
                "title": raw.get("title", ""),
                "description": raw.get("description", ""),
                "domain": raw.get("domain", task_dir.parent.name),
                "level": raw.get("level", ""),
                "timeout": int(raw.get("timeout", 3600)),
                "task_dir": str(task_dir),
            }
        )

    return tasks


def run_pinchbench(
    model_key: str,
    config: ModelConfig,
    *,
    clawevalkit_dir: Path,
    sample: int = 0,
    parallel: int = 1,
    results_dir: Path | None = None,
    force: bool = False,
    task_ids: list[str] | None = None,
) -> dict:
    """Run PinchBench evaluation using SAL's Bash Agent.

    Args:
        model_key: Model identifier
        config: Model configuration
        clawevalkit_dir: Path to ClawEvalkit root
        sample: Number of tasks to sample (0 = all)
        parallel: Parallel execution count
        results_dir: Results output directory
        force: Re-run even if cached results exist
        task_ids: Filter to specific task IDs

    Returns:
        Summary dict with score, passed, total, details
    """
    if results_dir is None:
        results_dir = clawevalkit_dir / "outputs" / "pinchbench"

    tasks = _load_pinchbench_tasks(clawevalkit_dir)
    if not tasks:
        return {"score": 0, "total": 0, "error": "no tasks loaded"}

    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    all_task_ids = [t["id"] for t in tasks]

    # Filter cached
    if not force:
        uncached = []
        for t in tasks:
            result_file = results_dir / model_key / t["id"] / "result.json"
            if not result_file.exists():
                uncached.append(t)
            else:
                try:
                    cached = json.loads(result_file.read_text())
                    if cached.get("status") != "success":
                        uncached.append(t)
                except Exception:
                    uncached.append(t)
        if uncached:
            print(
                f"[pinchbench] {len(tasks) - len(uncached)} cached, {len(uncached)} remaining"
            )
        tasks = uncached

    if sample and sample < len(tasks):
        random.seed(42)
        tasks = random.sample(tasks, sample)

    print(f"[pinchbench] Running {len(tasks)} tasks with model={model_key}")

    results = []
    if parallel <= 1:
        for i, task in enumerate(tasks):
            print(f"[pinchbench] ({i + 1}/{len(tasks)}) {task['id']}")
            start = time.time()
            r = run_pinchbench_task(task, config, results_dir)
            elapsed = time.time() - start
            print(
                f"[pinchbench]   -> {r['status']} mean={r['mean']:.4f} ({elapsed:.1f}s)"
            )
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(run_pinchbench_task, t, config, results_dir): t["id"]
                for t in tasks
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    print(f"[pinchbench] {tid} -> {r['status']} mean={r['mean']:.4f}")
                except Exception as e:
                    results.append(
                        {
                            "task_id": tid,
                            "status": "error",
                            "error": str(e),
                            "mean": 0.0,
                            "scores": {},
                        }
                    )
                    print(f"[pinchbench] {tid} -> error: {e}")

    # Load cached results too
    all_results = list(results)
    for t_id in all_task_ids:
        cached_file = results_dir / model_key / t_id / "result.json"
        if cached_file.exists() and not any(r.get("task_id") == t_id for r in results):
            try:
                cached = json.loads(cached_file.read_text())
                cached["_from_cache"] = True
                all_results.append(cached)
            except Exception:
                pass

    # Compute summary
    scored = [r for r in all_results if r.get("status") == "success"]
    means = [r["mean"] for r in scored if r.get("mean") is not None]
    overall = round(sum(means) / len(means) * 100, 1) if means else 0

    summary = {
        "model": model_key,
        "score": overall,
        "passed": len(means),
        "scored": len(scored),
        "total": len(all_task_ids),
        "pending": len(all_task_ids) - len(all_results),
        "details": all_results,
    }

    # Save summary
    summary_path = results_dir / f"{model_key}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Save without details for the summary file
    summary_save = {k: v for k, v in summary.items() if k != "details"}
    summary_save["result_count"] = len(all_results)
    summary_path.write_text(
        json.dumps(summary_save, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[pinchbench] Summary saved to {summary_path}")
    print(f"[pinchbench] Score: {overall} ({len(means)}/{len(all_task_ids)})")

    return summary


def run_agentbench(
    model_key: str,
    config: ModelConfig,
    *,
    clawevalkit_dir: Path,
    sample: int = 0,
    parallel: int = 1,
    results_dir: Path | None = None,
    force: bool = False,
    task_ids: list[str] | None = None,
) -> dict:
    """Run AgentBench evaluation using SAL's Bash Agent.

    Args:
        model_key: Model identifier
        config: Model configuration
        clawevalkit_dir: Path to ClawEvalkit root
        sample: Number of tasks to sample (0 = all)
        parallel: Parallel execution count
        results_dir: Results output directory
        force: Re-run even if cached results exist
        task_ids: Filter to specific task IDs

    Returns:
        Summary dict with score, passed, total, details
    """
    if results_dir is None:
        results_dir = clawevalkit_dir / "outputs" / "agentbench"

    tasks = _load_agentbench_tasks(clawevalkit_dir)
    if not tasks:
        return {"score": 0, "total": 0, "error": "no tasks loaded"}

    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]

    all_task_ids = [t["task_id"] for t in tasks]

    # Filter cached
    if not force:
        uncached = []
        for t in tasks:
            result_file = results_dir / model_key / t["task_id"] / "result.json"
            if not result_file.exists():
                uncached.append(t)
            else:
                try:
                    cached = json.loads(result_file.read_text())
                    if cached.get("status") != "success":
                        uncached.append(t)
                except Exception:
                    uncached.append(t)
        if uncached:
            print(
                f"[agentbench] {len(tasks) - len(uncached)} cached, {len(uncached)} remaining"
            )
        tasks = uncached

    if sample and sample < len(tasks):
        random.seed(42)
        tasks = random.sample(tasks, sample)

    print(f"[agentbench] Running {len(tasks)} tasks with model={model_key}")

    results = []
    if parallel <= 1:
        for i, task in enumerate(tasks):
            print(
                f"[agentbench] ({i + 1}/{len(tasks)}) {task['task_id']} [{task['category']}]"
            )
            start = time.time()
            r = run_agentbench_task(task, config, results_dir, clawevalkit_dir)
            elapsed = time.time() - start
            scores = r.get("scores", {})
            print(
                f"[agentbench]   -> {r['status']} overall={scores.get('overall_score', 'N/A')} ({elapsed:.1f}s)"
            )
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    run_agentbench_task, t, config, results_dir, clawevalkit_dir
                ): t["task_id"]
                for t in tasks
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    print(f"[agentbench] {tid} -> {r['status']}")
                except Exception as e:
                    results.append(
                        {
                            "task_id": tid,
                            "status": "error",
                            "error": str(e),
                            "scores": {},
                        }
                    )
                    print(f"[agentbench] {tid} -> error: {e}")

    # Compute summary
    scored = [r for r in results if r.get("status") == "success" and r.get("scores")]
    scores_list = [
        r["scores"]["overall_score"] for r in scored if "overall_score" in r["scores"]
    ]
    avg = round(sum(scores_list) / len(scores_list), 1) if scores_list else 0

    summary = {
        "model": model_key,
        "score": avg,
        "passed": len(scores_list),
        "total": len(all_task_ids),
        "details": results,
    }

    summary_path = results_dir / f"{model_key}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_save = {k: v for k, v in summary.items() if k != "details"}
    summary_save["result_count"] = len(results)
    summary_path.write_text(
        json.dumps(summary_save, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[agentbench] Summary saved to {summary_path}")
    print(f"[agentbench] Score: {avg} ({len(scores_list)}/{len(all_task_ids)})")

    return summary


def run_clawbench(
    model_key: str,
    config: ModelConfig,
    *,
    clawevalkit_dir: Path,
    sample: int = 0,
    parallel: int = 1,
    results_dir: Path | None = None,
    force: bool = False,
    task_ids: list[str] | None = None,
) -> dict:
    """Run ClawBench Official using SAL's Bash Agent and pytest verifiers."""
    if results_dir is None:
        results_dir = clawevalkit_dir / "outputs" / "clawbench-official"

    tasks = _load_clawbench_tasks(clawevalkit_dir)
    if not tasks:
        return {"score": 0, "total": 0, "error": "no tasks loaded"}

    if task_ids:
        task_id_set = set(task_ids)
        tasks = [
            t
            for t in tasks
            if t["id"] in task_id_set or t.get("dir_name") in task_id_set
        ]

    all_task_ids = [t["id"] for t in tasks]

    if not force:
        uncached = []
        for t in tasks:
            result_file = results_dir / model_key / t["id"] / "result.json"
            if not result_file.exists():
                uncached.append(t)
            else:
                try:
                    cached = json.loads(result_file.read_text())
                    if cached.get("status") != "success":
                        uncached.append(t)
                except Exception:
                    uncached.append(t)
        if uncached:
            print(
                f"[clawbench] {len(tasks) - len(uncached)} cached, {len(uncached)} remaining"
            )
        tasks = uncached

    if sample and sample < len(tasks):
        random.seed(42)
        tasks = random.sample(tasks, sample)

    print(f"[clawbench] Running {len(tasks)} tasks with model={model_key}")

    results = []
    if parallel <= 1:
        for i, task in enumerate(tasks):
            print(
                f"[clawbench] ({i + 1}/{len(tasks)}) {task['id']} [{task.get('domain', '')}]"
            )
            start = time.time()
            r = run_clawbench_task(task, config, results_dir)
            elapsed = time.time() - start
            print(
                f"[clawbench]   -> {r['status']} score={r.get('score', 0):.4f} ({elapsed:.1f}s)"
            )
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(run_clawbench_task, t, config, results_dir): t["id"]
                for t in tasks
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    print(
                        f"[clawbench] {tid} -> {r['status']} score={r.get('score', 0):.4f}"
                    )
                except Exception as e:
                    results.append(
                        {
                            "task_id": tid,
                            "status": "error",
                            "error": str(e),
                            "score": 0.0,
                        }
                    )
                    print(f"[clawbench] {tid} -> error: {e}")

    all_results = list(results)
    for t_id in all_task_ids:
        cached_file = results_dir / model_key / t_id / "result.json"
        if cached_file.exists() and not any(r.get("task_id") == t_id for r in results):
            try:
                cached = json.loads(cached_file.read_text())
                cached["_from_cache"] = True
                all_results.append(cached)
            except Exception:
                pass

    scored = [r for r in all_results if r.get("status") == "success"]
    scores = [r.get("score", 0.0) for r in scored]
    overall = round(sum(scores) / len(scores) * 100, 1) if scores else 0

    summary = {
        "model": model_key,
        "score": overall,
        "passed": sum(1 for r in all_results if r.get("passed")),
        "scored": len(scored),
        "total": len(all_task_ids),
        "pending": len(all_task_ids) - len(all_results),
        "details": all_results,
    }

    summary_path = results_dir / f"{model_key}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_save = {k: v for k, v in summary.items() if k != "details"}
    summary_save["result_count"] = len(all_results)
    summary_path.write_text(
        json.dumps(summary_save, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[clawbench] Summary saved to {summary_path}")
    print(f"[clawbench] Score: {overall} ({len(scores)}/{len(all_task_ids)})")

    return summary


BENCHMARK_RUNNERS = {
    "pinchbench": run_pinchbench,
    "agentbench": run_agentbench,
    "clawbench": run_clawbench,
    "clawbench-official": run_clawbench,
}


def run_eval(
    bench_key: str,
    model_key: str,
    *,
    clawevalkit_dir: str | Path | None = None,
    sample: int = 0,
    parallel: int = 1,
    results_dir: str | Path | None = None,
    force: bool = False,
    task_ids: list[str] | None = None,
    max_turns: int = 20,
    timeout: int = 300,
    api_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    provider: str | None = None,
) -> dict:
    """Run a ClawEvalkit benchmark evaluation using Simple Agent Lab.

    Args:
        bench_key: Benchmark key ("pinchbench" or "agentbench")
        model_key: Model identifier
        clawevalkit_dir: Path to a benchmark root. For ClawBench this can be
            the repo-local assets/benchmarks/claw-bench directory and does not
            require Docker.
        sample: Number of tasks to sample (0 = all)
        parallel: Parallel task execution count
        results_dir: Results output directory
        force: Re-run cached results
        task_ids: Filter to specific task IDs
        max_turns: Max agent turns per task
        timeout: Per-task timeout in seconds
        api_url: Override API base URL
        api_key: Override API key directly
        api_key_env: Override env var name for API key
        provider: Override provider kind

    Returns:
        Summary dict
    """
    if clawevalkit_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        if bench_key in {"clawbench", "clawbench-official"}:
            clawevalkit_dir = repo_root / "assets" / "benchmarks" / "claw-bench"
        else:
            clawevalkit_dir = repo_root / "clawevalkit_dir"
    clawevalkit_dir = Path(clawevalkit_dir)

    if not clawevalkit_dir.exists():
        return {
            "score": 0,
            "total": 0,
            "error": f"Benchmark dir not found: {clawevalkit_dir}",
        }

    if bench_key not in BENCHMARK_RUNNERS:
        return {
            "score": 0,
            "total": 0,
            "error": f"Unknown benchmark: {bench_key}. Supported: {list(BENCHMARK_RUNNERS.keys())}",
        }

    config = build_model_config(
        model_key,
        clawevalkit_dir=clawevalkit_dir,
        api_url=api_url,
        api_key_env=api_key_env,
        provider=provider,
        max_turns=max_turns,
        timeout=timeout,
    )

    print(f"[openclaw] Benchmark: {bench_key}")
    print(f"[openclaw] Model: {config.name} ({config.model})")
    print(f"[openclaw] Provider: {config.api_kind} @ {config.api_url}")
    print(f"[openclaw] Key env: {config.api_key_env}")
    print(f"[openclaw] Max turns: {config.max_turns}, Timeout: {config.timeout}s")

    runner = BENCHMARK_RUNNERS[bench_key]
    return runner(
        model_key,
        config,
        clawevalkit_dir=clawevalkit_dir,
        sample=sample,
        parallel=parallel,
        results_dir=Path(results_dir) if results_dir else None,
        force=force,
        task_ids=task_ids,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ClawEvalkit benchmarks with Simple Agent Lab's Bash Agent"
    )
    parser.add_argument(
        "--bench",
        required=True,
        choices=list(BENCHMARK_RUNNERS.keys()),
        help="Benchmark to run",
    )
    parser.add_argument(
        "--model", required=True, help="Model key (e.g. claude-sonnet, glm-4.7)"
    )
    parser.add_argument(
        "--clawevalkit",
        default=None,
        help=(
            "Path to ClawEvalkit root or claw-bench root. For ClawBench, "
            "defaults to repo-local assets/benchmarks/claw-bench."
        ),
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Number of tasks to sample (0 = all)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Parallel task execution count"
    )
    parser.add_argument("--results-dir", default=None, help="Results output directory")
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if cached results exist"
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None, help="Filter to specific task IDs"
    )
    parser.add_argument(
        "--max-turns", type=int, default=20, help="Max agent turns per task"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-task timeout in seconds"
    )
    parser.add_argument("--api-url", default=None, help="Override API base URL")
    parser.add_argument("--api-key", default=None, help="Override API key directly")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Env var name for API key (default: auto-detect from provider)",
    )
    parser.add_argument("--provider", default=None, help="Override provider kind")

    args = parser.parse_args()

    result = run_eval(
        bench_key=args.bench,
        model_key=args.model,
        clawevalkit_dir=args.clawevalkit,
        sample=args.sample,
        parallel=args.parallel,
        results_dir=args.results_dir,
        force=args.force,
        task_ids=args.task_ids,
        max_turns=args.max_turns,
        timeout=args.timeout,
        api_url=args.api_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        provider=args.provider,
    )

    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
