"""SkillsBench container half: build task + extract result + evaluate via pytest.

The agent solves a coding/algorithm task in the workspace. The evaluate hook
runs pytest with the task's test_outputs.py against the workspace to score.

Overall logic:
1. build_task: seeds env files, fixes paths, returns the instruction
2. extract_result: lists workspace files produced by the agent
3. evaluate: runs pytest with the task's test code (paths fixed) against
   the workspace, collecting pass/fail and computing a percentage score
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="skillsbench_agent",
        role="Expert programmer solving algorithmic and engineering tasks.",
        system_prompt=(
            "You are an expert programmer. Solve the task by writing code and "
            "producing the required output files in the current workspace directory. "
            "Ensure your solution is correct, handles edge cases, and produces "
            "output in the exact format specified."
        ),
        flavor="bash_task",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the agent-visible task from the instance data.

    Seeds environment files into the workspace and fixes hardcoded paths
    in the instruction text (e.g. /root/ → workspace/).
    """
    workspace = Path(workdir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy environment files from data dir to workspace
    data_dir = instance.get("data_dir", "")
    env_files = instance.get("env_files", [])
    if data_dir and env_files:
        data_path = Path(data_dir)
        for rel_path in env_files:
            src = data_path / "tasks" / instance["instance_id"] / rel_path
            if src.exists():
                fname = Path(rel_path).name
                dst = workspace / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

    # Fix instruction paths: replace /root/ references with workspace-relative paths
    instruction = str(instance["instruction"])
    instruction = instruction.replace("/root/", "./")

    return instruction


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """List workspace files produced by the agent."""
    files = []
    for f in sorted(workspace.rglob("*")):
        if f.is_file() and not f.name.startswith("_"):
            files.append(str(f.relative_to(workspace)))
    return {
        "instance_id": instance["instance_id"],
        "workspace_files": files,
        "status": "completed",
    }


def evaluate(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the task's pytest tests against the workspace.

    SkillsBench tests typically use a TestOutputs class with methods
    that check for output files and validate their contents. We run
    pytest pointing to the agent's workspace.
    """
    eval_data = (context or {}).get("eval", {})
    instance_id = eval_data.get("instance_id", instance.get("instance_id", ""))
    test_code = eval_data.get("test_code", "")
    timeout = eval_data.get("verifier_timeout", 900)

    if not test_code:
        return {
            "instance_id": instance_id,
            "score": 0,
            "error": "no test code",
            "tests_passed": 0,
            "tests_total": 0,
        }

    with tempfile.TemporaryDirectory(prefix="skillsbench_") as tmp:
        # Fix hardcoded paths in test code: /root/ → actual workspace
        fixed_test_code = test_code.replace("/root/", str(workspace) + "/")

        test_file = Path(tmp) / "test_outputs.py"
        test_file.write_text(fixed_test_code)

        try:
            result = subprocess.run(
                [
                    "python3", "-m", "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                    "--no-header",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=min(timeout, 120),
                cwd=tmp,
                env={
                    **_get_env(),
                    "WORKSPACE": str(workspace),
                },
            )
        except subprocess.TimeoutExpired:
            return {
                "instance_id": instance_id,
                "score": 0,
                "error": "pytest timeout",
                "tests_passed": 0,
                "tests_total": 0,
            }
        except Exception as exc:
            return {
                "instance_id": instance_id,
                "score": 0,
                "error": f"pytest error: {exc}",
                "tests_passed": 0,
                "tests_total": 0,
            }

    output = result.stdout + result.stderr
    passed, failed, errors = _parse_pytest_output(output)
    total = passed + failed + errors

    score = round((passed / total) * 100, 1) if total > 0 else 0

    return {
        "instance_id": instance_id,
        "score": score,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "tests_total": total,
        "pytest_exit_code": result.returncode,
        "pytest_output": output[-2000:] if len(output) > 2000 else output,
    }


def _get_env() -> dict[str, str]:
    """Get current environment variables as a dict."""
    import os
    return dict(os.environ)


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest output to count passed, failed, error tests."""
    passed = failed = errors = 0

    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        errors = int(m.group(1))

    if passed or failed or errors:
        return passed, failed, errors

    # Fallback: count per-test lines
    for line in output.split("\n"):
        stripped = line.strip()
        if "PASSED" in stripped:
            passed += 1
        elif "FAILED" in stripped:
            failed += 1
        elif "ERROR" in stripped:
            errors += 1

    return passed, failed, errors
