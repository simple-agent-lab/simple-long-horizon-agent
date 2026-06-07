"""ClawBench-Official container half: build task + extract result + evaluate via pytest.

The agent works in /workspace to complete the task described in instruction.md.
The evaluate hook runs the verifier's pytest tests against the workspace to score it.

Overall logic:
1. build_task: returns the instruction.md text as the agent's task prompt
2. extract_result: scans the workspace for output files the agent created
3. evaluate: runs pytest with the verifier tests against the workspace,
   collecting pass/fail per test and computing a weighted score
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="clawbench_official_agent",
        role="Expert agent solving real-world tasks in a workspace.",
        system_prompt=(
            "You are an expert agent working in a workspace directory. "
            "Read the task instruction carefully and produce all required output files. "
            "Run all processes in the foreground without user input. "
            "Write all output files to the current workspace directory. "
            "Ensure your outputs are complete, correct, and well-formatted."
        ),
        flavor="bash_task",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the agent-visible task from the instance data.

    Returns the instruction.md content. Also creates the workspace directory
    structure and seeds any environment data files if present.
    """
    workspace = Path(workdir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Run setup script if provided (creates workspace dir, seeds data)
    setup_script = instance.get("setup_script", "")
    if setup_script:
        setup_path = workspace / "_setup.sh"
        setup_path.write_text(setup_script)
        try:
            subprocess.run(
                ["bash", str(setup_path), str(workspace)],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass

    # Fix instruction paths: ClawEvalkit uses "workspace/..." as prefix
    # but in simple-agent-lab the workdir IS the workspace.
    instruction = str(instance["instruction"])
    instruction = instruction.replace("`workspace/", "`./")
    instruction = instruction.replace("workspace/", "")

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
    """Run the verifier pytest tests against the workspace.

    Writes the verifier test code to a temp file and runs pytest with
    --workspace pointing to the agent's output directory. Collects
    pass/fail counts and computes a score.
    """
    eval_data = (context or {}).get("eval", {})
    instance_id = eval_data.get("instance_id", instance.get("instance_id", ""))
    verifier_code = eval_data.get("verifier_code", "")

    if not verifier_code:
        return {
            "instance_id": instance_id,
            "score": 0,
            "error": "no verifier code",
            "tests_passed": 0,
            "tests_total": 0,
        }

    # Write verifier tests to a temp directory
    with tempfile.TemporaryDirectory(prefix="cb_official_") as tmp:
        test_file = Path(tmp) / "test_verifier.py"
        test_file.write_text(verifier_code)

        # Write conftest.py to register --workspace option
        conftest = Path(tmp) / "conftest.py"
        conftest.write_text(textwrap.dedent("""\
            import pytest
            from pathlib import Path

            def pytest_addoption(parser):
                parser.addoption("--workspace", default=".")

            @pytest.fixture
            def workspace(request):
                ws = request.config.getoption("--workspace")
                if ws:
                    return Path(ws)
                return Path(".")
        """))

        # Run pytest with --workspace flag and WORKSPACE env var
        try:
            result = subprocess.run(
                [
                    "python3", "-m", "pytest",
                    str(test_file),
                    "--workspace", str(workspace),
                    "-v",
                    "--tb=short",
                    "--no-header",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tmp,
                env={**os.environ, "WORKSPACE": str(workspace)},
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

    # Parse pytest output for pass/fail counts
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


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest verbose output to count passed, failed, and error tests.

    Looks for patterns like:
      PASSED / FAILED / ERROR per test line
      Summary line: "X passed, Y failed, Z errors"
    """
    import re

    passed = failed = errors = 0

    # Try summary line first: "3 passed, 1 failed, 1 error"
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

    # Fallback: count PASSED/FAILED/ERROR in per-test lines
    for line in output.split("\n"):
        stripped = line.strip()
        if "PASSED" in stripped:
            passed += 1
        elif "FAILED" in stripped:
            failed += 1
        elif "ERROR" in stripped:
            errors += 1

    return passed, failed, errors
