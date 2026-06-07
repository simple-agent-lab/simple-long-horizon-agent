"""PinchBench container half: agent executes task, evaluate runs grade() function.

The agent runs in the workspace, then the evaluate hook executes the embedded
grade(transcript, workspace_path) function to compute scores.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        name="pinchbench_agent",
        role="Expert agent solving tasks in a workspace.",
        system_prompt=(
            "You are an expert agent working in a restricted environment. "
            "Solve the task efficiently. Run all processes in the foreground "
            "without user input. Provide a complete, functional solution. "
            "Write all output files to the current workspace directory."
        ),
        flavor="bash",
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    return str(instance["prompt"])


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # List workspace files
    files = []
    for f in workspace.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(workspace)))
    return {"instance_id": instance["instance_id"], "workspace_files": files}


def evaluate(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the embedded grade() function to score the task.

    The grade function receives the trajectory (as a list of dicts) and the
    workspace path. It returns a dict of criterion → score (0-1).
    """
    eval_data = (context or {}).get("eval", {})
    grade_code = eval_data.get("grade_code", "")
    if not grade_code:
        return {"instance_id": eval_data.get("instance_id", ""), "score": 0, "error": "no grade code"}

    instance_id = eval_data.get("instance_id", instance.get("instance_id", ""))

    # Prefer the live trace passed by the generic runner. In the normal
    # ArtifactStore layout, trajectory.jsonl lives in the run directory rather
    # than under the task workspace.
    trajectory_jsonl = (context or {}).get("trajectory_jsonl", "")
    if trajectory_jsonl:
        transcript = _build_transcript_from_jsonl(str(trajectory_jsonl))
    else:
        trajectory_path = workspace / "out" / "trajectory.jsonl"
        transcript = _build_transcript(trajectory_path)

    # Execute grade function
    scores = _run_grade(grade_code, transcript, str(workspace))

    if scores:
        mean_score = sum(scores.values()) / len(scores)
    else:
        mean_score = 0.0

    return {
        "instance_id": instance_id,
        "scores": scores,
        "mean": round(mean_score, 4),
        "score": round(mean_score * 100, 1),
    }


def _build_transcript(trajectory_path: Path) -> list[dict]:
    """Convert trajectory.jsonl to a transcript format the grade() functions expect.

    The grade() functions from ClawEvalkit expect a list of event dicts with
    type "message" containing {message: {role, content}} or tool calls.
    """
    if not trajectory_path.exists():
        return []

    return _build_transcript_from_jsonl(trajectory_path.read_text())


def _build_transcript_from_jsonl(trajectory_jsonl: str) -> list[dict]:
    """Convert serialized trajectory JSONL into the grade() transcript shape."""
    transcript = []
    for line in trajectory_jsonl.strip().split("\n"):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        for msg in entry.get("messages", []):
            role = msg.get("role", "")
            kind = msg.get("kind", "")
            content = msg.get("content", [])

            if role == "assistant" and kind in ("step", "final"):
                # Convert content blocks to NanoBot-style transcript
                if isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("kind") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("kind") == "tool_call":
                                tool_calls.append({
                                    "name": block.get("name", ""),
                                    "arguments": block.get("arguments", {}),
                                })
                    if tool_calls:
                        transcript.append({
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "toolCall", "name": tc["name"], "arguments": tc["arguments"]}
                                    for tc in tool_calls
                                ],
                            },
                        })
                    if text_parts:
                        transcript.append({
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": " ".join(text_parts),
                            },
                        })
                elif isinstance(content, str):
                    transcript.append({
                        "type": "message",
                        "message": {"role": "assistant", "content": content},
                    })

            elif role == "user" and kind == "tool_result":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("kind") == "tool_result":
                            inner = block.get("content", [])
                            texts = [
                                b.get("text", "") for b in inner
                                if isinstance(b, dict) and b.get("kind") == "text"
                            ]
                            transcript.append({
                                "type": "message",
                                "message": {
                                    "role": "toolResult",
                                    "content": [{"type": "text", "text": t} for t in texts],
                                },
                            })

            elif role == "user" and kind == "task":
                if isinstance(content, list):
                    texts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("kind") == "text"
                    ]
                    if texts:
                        transcript.append({
                            "type": "message",
                            "message": {"role": "user", "content": " ".join(texts)},
                        })

    return transcript


def _run_grade(grade_code: str, transcript: list[dict], workspace_path: str) -> dict:
    """Execute the embedded grade(transcript, workspace_path) function."""
    namespace: dict[str, Any] = {}
    try:
        exec(grade_code, namespace)
        if "grade" in namespace and callable(namespace["grade"]):
            return namespace["grade"](transcript, workspace_path)
    except Exception:
        pass
    return {}
