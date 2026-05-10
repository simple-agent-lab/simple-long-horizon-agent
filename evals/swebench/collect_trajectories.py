"""Collect SWE-bench adapter trajectories and prediction JSONL.

Run from the repo root:

    PYTHONPATH=src python3 evals/swebench/collect_trajectories.py --allow-empty-patch

This script is a suite-specific adapter. It produces the project-owned raw
trajectory record plus the official SWE-bench prediction shape, but it does not
score the run.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.bash_tool import make_bash_tool
from simple_agent_lab.core import (
    Agent,
    AgentRuntime,
    Event,
    State,
    make_llm_step,
    run,
    sequence,
    until_final,
)
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.messages import Message, assistant_message
from simple_agent_lab.trajectory import (
    ModelTurn,
    RunTrace,
    json_safe,
    trace_record,
    write_jsonl,
)


DEFAULT_INSTANCE_ID = "sympy__sympy-20590"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL_NAME = "simple-agent-lab-smoke"
DEFAULT_WORKSPACE_COMMAND = "git diff --src-prefix=a/ --dst-prefix=b/"
DEFAULT_PROBLEM_STATEMENT = (
    "Smoke setup placeholder. Replace this with a real SWE-bench "
    "problem_statement before running the official harness."
)


def load_instance(path: str | None, instance_id: str | None) -> dict[str, Any]:
    if path is None:
        return {
            "instance_id": instance_id or DEFAULT_INSTANCE_ID,
            "repo": "sympy/sympy",
            "problem_statement": DEFAULT_PROBLEM_STATEMENT,
        }

    records = _load_instance_records(Path(path))
    if not records:
        raise SystemExit(f"No instance records found in {path}")
    if instance_id is None:
        return dict(records[0])
    for record in records:
        if str(record.get("instance_id")) == instance_id:
            return dict(record)
    raise SystemExit(f"Instance {instance_id!r} not found in {path}")


def _load_instance_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if "instances" in data and isinstance(data["instances"], list):
            return [dict(item) for item in data["instances"]]
        return [dict(data)]
    raise SystemExit(f"Unsupported instance record shape in {path}")


def load_patch(args: argparse.Namespace) -> tuple[str, str]:
    if args.patch_file and args.model_patch is not None:
        raise SystemExit("Use either --patch-file or --model-patch, not both.")
    if args.patch_file:
        return normalize_model_patch(Path(args.patch_file).read_text(encoding="utf-8")), "patch-file"
    if args.model_patch is not None:
        return normalize_model_patch(args.model_patch), "argument"
    return "", "empty"


def normalize_model_patch(text: str) -> str:
    """Return a unified diff from raw model text when it is easy to identify."""

    stripped = text.strip()
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if not stripped:
        return ""
    if stripped.startswith("diff --git ") or stripped.startswith("--- "):
        return stripped + "\n"
    return stripped + "\n"


def task_from_instance(instance: dict[str, Any]) -> str:
    problem = str(
        instance.get("problem_statement")
        or instance.get("problem")
        or instance.get("description")
        or ""
    )
    lines = [
        "Solve this SWE-bench instance. Return only a unified diff patch.",
        "",
        f"instance_id: {instance.get('instance_id', '')}",
    ]
    if instance.get("repo"):
        lines.append(f"repo: {instance['repo']}")
    if instance.get("base_commit"):
        lines.append(f"base_commit: {instance['base_commit']}")
    lines.extend(["", "problem_statement:", problem])
    return "\n".join(lines)


def make_patch_agent(patch: str) -> Agent:
    def step(agent: Agent, visible: list[Message], state: State) -> Message:
        del visible, state
        return assistant_message(
            patch,
            sender=agent.name,
            target="user",
            kind="final",
        )

    return Agent(
        name="swebench_agent",
        role="Read a SWE-bench issue and return only a unified diff patch.",
        step=step,
    )


def make_workspace_agent() -> Agent:
    return Agent(
        name="swebench_agent",
        role=(
            "Work in the prepared SWE-bench repository. Use bash for local "
            "inspection or edits, then return a concise final note."
        ),
        step=make_llm_step(
            LLMProvider(id="fake", api="fake", model="fake-model"),
            system_prompt=(
                "You are a tiny SWE-bench workspace agent. If the task names a "
                "bash command, call the bash tool once. After the tool result, "
                "return a short final answer."
            ),
            target="user",
        ),
    )


def run_patch_agent(instance: dict[str, Any], patch: str) -> State:
    state = State(
        task=task_from_instance(instance),
        data={
            "suite": "swebench",
            "instance": instance,
            "model_patch": patch,
        },
    )
    state.send("task", "user", "swebench_agent", state.task)
    for _ in run([make_patch_agent(patch)], state, sequence("swebench_agent")):
        pass
    return state


def run_workspace_agent(
    *,
    instance: dict[str, Any],
    workspace: Path,
    command: str,
    max_turns: int,
) -> State:
    repo_dir = resolve_repo_dir(workspace)
    task = (
        task_from_instance(instance)
        + "\n\n"
        + "For this setup run, use bash command: "
        + f"`{command}`"
    )
    runtime = AgentRuntime(
        [make_workspace_agent()],
        tools=[make_bash_tool(cwd=repo_dir)],
    )
    for _ in runtime.prompt(
        task,
        target="swebench_agent",
        next_agent=until_final("swebench_agent", max_turns=max_turns),
    ):
        pass
    runtime.state.data.update({
        "suite": "swebench",
        "instance": instance,
        "workspace": str(repo_dir),
        "model_patch": git_diff(repo_dir),
    })
    return runtime.state


def resolve_repo_dir(path: Path) -> Path:
    resolved = path.resolve()
    if (resolved / ".git").exists():
        return resolved
    nested = resolved / "repo"
    if (nested / ".git").exists():
        return nested
    return resolved


def git_diff(repo_dir: Path) -> str:
    if not (repo_dir / ".git").exists():
        return ""
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=repo_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    completed = subprocess.run(
        ["git", "diff", "--src-prefix=a/", "--dst-prefix=b/"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return normalize_model_patch(completed.stdout)


def model_turns_from_events(trace_id: str, events: list[Event]) -> list[ModelTurn]:
    turns: list[ModelTurn] = []
    pending: dict[str, Any] | None = None
    model_call_index = 0

    for event in events:
        if event.kind == "model_request":
            model_call_index += 1
            pending = {
                "agent": str(event.data.get("agent") or ""),
                "input_messages": event.data.get("llm_payload") or [],
                "tools": event.data.get("tools") or [],
                "request_event_index": event.index,
                "meta": {
                    "visible_count": event.data.get("visible_count"),
                    "model_message_count": event.data.get("llm_message_count"),
                },
            }
            continue

        if event.kind != "message" or pending is None:
            continue
        message = event.message
        if message is None or message.role != "assistant":
            continue
        agent = pending["agent"] or message.sender
        if message.sender != agent:
            continue
        turns.append(
            ModelTurn(
                step_id=f"{trace_id}.model{model_call_index}",
                agent=agent,
                input_messages=json_safe(pending["input_messages"]),
                output_message=json_safe(message),
                tools=json_safe(pending["tools"]),
                meta={
                    **pending["meta"],
                    "request_event_index": pending["request_event_index"],
                    "message_event_index": event.index,
                },
            )
        )
        pending = None

    return turns


def prediction_record(instance_id: str, model_name: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def trace_from_state(
    *,
    state: State,
    instance: dict[str, Any],
    dataset_name: str,
    split: str,
    model_name: str,
    patch_source: str,
) -> RunTrace:
    instance_id = str(instance["instance_id"])
    trace_id = f"swebench.{instance_id}"
    return RunTrace(
        trace_id=trace_id,
        producer="suite:swebench",
        task=state.task,
        messages=json_safe(state.messages),
        events=json_safe(state.events),
        model_turns=model_turns_from_events(trace_id, state.events),
        meta={
            "suite": "swebench",
            "dataset_name": dataset_name,
            "split": split,
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "patch_source": patch_source,
            "patch_chars": len(str(state.data.get("model_patch") or "")),
            "workspace": state.data.get("workspace"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-json", help="SWE-bench instance JSON or JSONL.")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--workspace",
        help=(
            "Prepared workspace or repo directory. If no patch is provided, "
            "the agent runs there and git diff becomes model_patch."
        ),
    )
    parser.add_argument(
        "--agent-command",
        default=DEFAULT_WORKSPACE_COMMAND,
        help="Bash command the setup agent should run in --workspace mode.",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--patch-file", help="File containing a candidate unified diff.")
    parser.add_argument("--model-patch", help="Candidate patch text.")
    parser.add_argument(
        "--allow-empty-patch",
        action="store_true",
        help="Allow a no-op prediction for adapter smoke tests.",
    )
    parser.add_argument(
        "--traces",
        default=str(ROOT / "evals/out/swebench_trajectories.jsonl"),
        help="Trajectory JSONL output.",
    )
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "evals/out/swebench_predictions.jsonl"),
        help="SWE-bench prediction JSONL output.",
    )
    args = parser.parse_args()

    instance = load_instance(args.instance_json, args.instance_id)
    use_workspace = bool(args.workspace and not args.patch_file and args.model_patch is None)
    if use_workspace:
        state = run_workspace_agent(
            instance=instance,
            workspace=Path(args.workspace),
            command=args.agent_command,
            max_turns=args.max_turns,
        )
        patch = str(state.data.get("model_patch") or "")
        patch_source = "workspace-diff"
    else:
        patch, patch_source = load_patch(args)
        state = run_patch_agent(instance, patch)

    if not patch.strip() and not args.allow_empty_patch:
        raise SystemExit("Empty patch. Pass --allow-empty-patch for local smoke setup.")

    trace = trace_from_state(
        state=state,
        instance=instance,
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        patch_source=patch_source,
    )
    prediction = prediction_record(str(instance["instance_id"]), args.model_name, patch)

    write_jsonl(args.traces, [trace_record(trace)])
    write_jsonl(args.predictions, [prediction])

    print(f"wrote 1 SWE-bench trajectory to {args.traces}")
    print(f"wrote 1 SWE-bench prediction to {args.predictions}")
    print(
        f"{trace.trace_id}: model_turns={len(trace.model_turns)} "
        f"patch_chars={len(patch)}"
    )


if __name__ == "__main__":
    main()
