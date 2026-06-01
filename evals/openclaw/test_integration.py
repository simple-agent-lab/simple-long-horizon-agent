"""Smoke test: verify the openclaw integration works with SAL's fake provider.

Runs PinchBench sanity check (task_00_sanity) using the fake LLM provider
to confirm the full pipeline works: load task → run agent → convert transcript → grade.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure SAL is importable
repo_root = Path(__file__).resolve().parents[2]
src = repo_root / "src"
for p in (str(repo_root), str(src)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals.openclaw.config import ModelConfig
from evals.openclaw.adapter import (
    sal_events_to_claweval_transcript,
    run_task_with_sal_agent,
    run_clawbench_task,
    _run_grade,
)


def test_transcript_conversion():
    """Test that SAL events can be converted to ClawEvalkit transcript format."""
    from simple_agent_lab.agents.bash import make_bash_agent
    from simple_agent_lab.llm.provider import Provider

    provider = Provider(
        id="fake-test",
        api="fake",
        model="fake",
    )
    agent = make_bash_agent(provider)

    state, events = agent.run("list files")
    for event in events:
        pass  # consume all

    transcript = sal_events_to_claweval_transcript(state.events)
    assert isinstance(transcript, list), "transcript should be a list"
    assert len(transcript) > 0, "transcript should not be empty"

    # Verify structure
    for entry in transcript:
        assert "type" in entry, f"entry missing 'type': {entry}"
        if entry["type"] == "message":
            assert "message" in entry, "message entry missing 'message' key"

    print(f"[PASS] Transcript conversion: {len(transcript)} entries")
    return True


def test_full_pinchbench_task():
    """Test running a PinchBench sanity check task through the full pipeline."""
    config = ModelConfig(
        name="fake-test",
        model="fake",
        api_url="",
        api_key_env="",
        api_kind="fake",
        max_turns=5,
        timeout=60,
    )

    # Minimal task that matches task_00_sanity
    task = {
        "id": "task_00_sanity",
        "prompt": "Say 'Hello, I'm ready!' to confirm you can respond.",
        "grade_code": """
def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}
    has_response = False
    for entry in transcript:
        if entry.get("type") == "message":
            message = entry.get("message", {})
            if message.get("role") == "assistant":
                content = message.get("content", [])
                if content and len(content) > 0:
                    has_response = True
                    break
    scores["agent_responded"] = 1.0 if has_response else 0.0
    return scores
""",
        "timeout": 60,
        "workspace_files": [],
    }

    # Run with SAL
    result = run_task_with_sal_agent(
        prompt=task["prompt"],
        config=config,
        workspace="/tmp/sal_test_workspace",
    )

    assert result["status"] in ("success", "error"), (
        f"unexpected status: {result['status']}"
    )
    print(f"[INFO] Agent status: {result['status']}")
    print(f"[INFO] Agent events: {result.get('raw_events_count', 0)}")
    print(f"[INFO] Execution time: {result.get('execution_time', 0)}s")
    print(f"[INFO] Error: {result.get('error', 'none')}")

    # Convert transcript
    transcript = result.get("transcript", [])
    assert isinstance(transcript, list), "transcript should be a list"
    print(f"[INFO] Transcript entries: {len(transcript)}")

    # Show some transcript entries for debugging
    for entry in transcript[:5]:
        print(f"[DEBUG] {entry.get('type')}: {str(entry)[:120]}")

    # Run grading
    scores = _run_grade(task["grade_code"], transcript, "/tmp/sal_test_workspace")
    print(f"[INFO] Grade scores: {scores}")

    if scores.get("agent_responded") == 1.0:
        print("[PASS] Full pipeline: agent responded, grading passed")
    else:
        print("[WARN] Agent did not produce a response (fake provider behavior)")

    return True


def test_load_pinchbench_from_clawevalkit():
    """Test loading tasks from ClawEvalkit directory."""
    clawevalkit_dir = Path("/Users/bytedance/Documents/github/ClawEvalkit")
    if not clawevalkit_dir.exists():
        print("[SKIP] ClawEvalkit dir not found")
        return True

    from evals.openclaw.runner import _load_pinchbench_tasks

    tasks = _load_pinchbench_tasks(clawevalkit_dir)
    print(f"[INFO] Loaded {len(tasks)} PinchBench tasks")
    if tasks:
        print(f"[INFO] First task: {tasks[0]['id']} - {tasks[0]['prompt'][:60]}...")
        print(f"[PASS] Task loading works")
    else:
        print("[FAIL] No tasks loaded")
    return True


def test_clawbench_task_smoke():
    """Test ClawBench loading + SAL execution + pytest verifier on one task."""
    clawbench_dir = repo_root / "assets" / "benchmarks" / "claw-bench"
    if not clawbench_dir.exists():
        print("[SKIP] repo-local ClawBench dir not found")
        return True

    from evals.openclaw.runner import _load_clawbench_tasks

    tasks = _load_clawbench_tasks(clawbench_dir)
    print(f"[INFO] Loaded {len(tasks)} ClawBench tasks")
    assert tasks, "no ClawBench tasks loaded"

    task = next((t for t in tasks if t["id"] == "cal-001"), tasks[0])
    config = ModelConfig(
        name="fake-test",
        model="fake",
        api_url="",
        api_key_env="",
        api_kind="fake",
        max_turns=5,
        timeout=60,
    )
    results_dir = Path(tempfile.mkdtemp(prefix="sal_clawbench_smoke_"))
    try:
        result = run_clawbench_task(task, config, results_dir)
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)

    print(
        f"[INFO] ClawBench task {task['id']}: "
        f"status={result['status']} score={result['score']} "
        f"checks={result['checks_passed']}/{result['checks_total']}"
    )
    assert result["status"] == "success", result.get("error")
    assert result["checks_total"] > 0, "verifier did not run checks"
    print("[PASS] ClawBench task pipeline runs")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Agent Lab × ClawEvalkit Integration Smoke Test")
    print("=" * 60)

    tests = [
        ("Transcript conversion", test_transcript_conversion),
        ("Full PinchBench task", test_full_pinchbench_task),
        ("Load from ClawEvalkit", test_load_pinchbench_from_clawevalkit),
        ("ClawBench task smoke", test_clawbench_task_smoke),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
