"""Adapter: Simple Agent Lab ↔ ClawEvalkit transcript format.

Converts SAL's runtime events (Event protocol) into ClawEvalkit's transcript
format (list of dicts with "type" + "message" fields), so that ClawEvalkit's
grading functions can score SAL agent runs.

Also provides task execution helpers that orchestrate: load task → run SAL agent
→ collect output → convert transcript → run grading.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import ModelConfig


def sal_events_to_claweval_transcript(events: list) -> list[dict]:
    """Convert SAL runtime events to ClawEvalkit transcript format.

    SAL events (frozen dataclasses from protocols.py) have fields like:
      - MessageEvent: event_type="message", message=Message(...)
      - ToolExecutionStartEvent: event_type="tool_execution_start", tool_name="bash"

    ClawEvalkit expects a flat list of dicts:
      {"type": "message", "message": {"role": "assistant/user/tool", "content": [...]}}
      {"type": "tool_call", "name": "bash", "params": {...}}
      {"type": "tool_result", "result": "..."}

    Args:
        events: List of SAL Event objects from run()

    Returns:
        List of ClawEvalkit-format transcript entries
    """
    transcript = []
    for event in events:
        try:
            etype = getattr(event, "event_type", None) or getattr(event, "kind", None)
            if etype is None:
                continue

            # Message events → transcript messages
            if etype == "message":
                msg = event.message
                role = msg.role
                content = _message_content_to_list(msg)
                transcript.append(
                    {
                        "type": "message",
                        "message": {
                            "role": role,
                            "content": content,
                        },
                    }
                )

            # Tool call events
            elif etype == "tool_execution_start":
                transcript.append(
                    {
                        "type": "tool_call",
                        "name": getattr(event, "tool_name", "unknown"),
                        "params": {},  # arguments may be in tool_execution_start or the assistant message
                    }
                )

            # Tool result events
            elif etype == "tool_execution_end":
                result_content = ""
                partial = getattr(event, "partial", None)
                if partial and hasattr(partial, "content"):
                    result_content = str(partial.content)
                transcript.append(
                    {
                        "type": "tool_result",
                        "result": result_content,
                    }
                )

        except Exception:
            continue

    return transcript


def _message_content_to_list(msg) -> list:
    """Convert SAL Message.content (tuple of ContentBlocks) to ClawEvalkit list format."""
    blocks = []
    for block in msg.content:
        block_type = getattr(block, "kind", None)

        if block_type == "text":
            blocks.append({"type": "text", "text": block.text})

        elif block_type == "tool_call":
            blocks.append(
                {
                    "type": "toolCall",
                    "name": block.name,
                    "arguments": dict(block.arguments) if block.arguments else {},
                    "id": getattr(block, "id", ""),
                }
            )

        elif block_type == "tool_result":
            inner_texts = []
            for inner in block.content:
                if hasattr(inner, "text"):
                    inner_texts.append(inner.text)
            blocks.append(
                {
                    "type": "toolResult",
                    "content": inner_texts,
                    "tool_call_id": block.tool_call_id,
                    "tool_name": block.tool_name,
                }
            )

        elif block_type == "thinking":
            blocks.append({"type": "thinking", "text": block.text})

        elif block_type == "image":
            blocks.append(
                {"type": "image", "data": block.data, "mime_type": block.mime_type}
            )

    return blocks


def _extract_tool_results_from_state(state) -> str:
    """Extract tool output text from SAL State for workspace inspection."""
    results = []
    for event in state.events:
        if hasattr(event, "kind") and event.kind == "message":
            msg = event.message
            if (
                msg.role == "user"
                and hasattr(msg, "kind")
                and msg.kind == "tool_result"
            ):
                for block in msg.content:
                    if hasattr(block, "content"):
                        for inner in block.content:
                            if hasattr(inner, "text"):
                                results.append(inner.text)
    return "\n".join(results)


def _build_azure_agent(config: ModelConfig, cwd: str | Path | None):
    """Build a SAL Agent backed by AzureOpenAI (for ModelHub / GPT proxy).

    SAL's built-in adapters don't support AzureOpenAI, so we construct
    the agent with a custom generate function that calls AzureOpenAI directly.
    """
    # Ensure SAL is importable
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    for p in (str(repo_root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from simple_agent_lab.core import Agent
    from simple_agent_lab.llm.bridge import (
        llm_response_to_assistant_message,
        messages_to_llm_messages,
        tool_to_llm_tool,
    )
    from simple_agent_lab.llm.adapters.openai_chat import (
        to_openai_chat_messages,
        to_openai_chat_tools,
    )
    from simple_agent_lab.llm.types import LLMMessage, LLMRequest
    from simple_agent_lab.llm.stream import complete as llm_complete
    from simple_agent_lab.messages import Message

    # Pre-load and cache the LLM message list + tools so we don't rebuild every turn
    _cache = {}

    # Clear proxy for internal APIs
    import os

    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[k] = ""

    api_key = os.environ.get(config.api_key_env, "")

    # Build tools BEFORE the closure so they're in scope
    from simple_agent_lab.agents.bash import make_bash_tool

    tools_list = [make_bash_tool(cwd=cwd)]
    tools_tuple = tuple(tools_list)

    def azure_generate(visible: list[Message]) -> Message:
        from simple_agent_lab.messages import TextBlock, ToolCallBlock
        from simple_agent_lab.llm.bridge import llm_response_to_assistant_message
        from simple_agent_lab.llm.types import StopReason, TokenUsage, LLMResponse

        llm_messages = messages_to_llm_messages(visible)
        tool_defs = [tool_to_llm_tool(t) for t in tools_tuple]

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-03-01-preview",
            azure_endpoint=config.api_url,
            timeout=60.0,
        )

        oai_msgs = to_openai_chat_messages(llm_messages)
        kwargs: dict = {
            "model": config.model,
            "messages": oai_msgs,
        }
        if tool_defs:
            kwargs["tools"] = to_openai_chat_tools(tool_defs)

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        # Build content blocks from response
        content_blocks = []
        if choice.message.content:
            content_blocks.append(TextBlock(text=choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append(
                    ToolCallBlock(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    )
                )

        stop_reason = "end_turn" if not choice.message.tool_calls else "tool_use"
        llm_response = LLMResponse(
            content=tuple(content_blocks),
            stop_reason=stop_reason,
            usage=TokenUsage(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
        )

        kind = "final" if stop_reason == "end_turn" else "thought"
        return llm_response_to_assistant_message(
            llm_response,
            sender="bash_agent",
            target="user",
            kind=kind,
        )

    return Agent(
        name="bash_agent",
        generate=azure_generate,
        role="Use bash for local commands, then summarize what you observed.",
        tools=tools_tuple,
    )


def run_task_with_sal_agent(
    prompt: str,
    config: ModelConfig,
    *,
    cwd: str | Path | None = None,
    workspace: str | Path | None = None,
) -> dict:
    """Run a single task using Simple Agent Lab's Bash Agent.

    Supports both standard SAL providers and AzureOpenAI (for ModelHub / GPT proxy).

    Args:
        prompt: Task prompt / instruction
        config: Model configuration
        cwd: Working directory for the bash tool
        workspace: Task workspace directory (created if None)

    Returns:
        Dict with keys: status, content, transcript, usage, execution_time, error
    """
    # Ensure we can import SAL
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    for p in (str(repo_root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from simple_agent_lab.agents.bash import make_bash_agent
    from simple_agent_lab.llm.provider import Provider

    # Azure OpenAI path (ModelHub / GPT proxy)
    if config.api_kind == "azure_openai":
        agent = _build_azure_agent(config, cwd=cwd or workspace)
    else:
        # Standard SAL path
        provider = Provider(
            id=config.name,
            api=config.api_kind,
            model=config.model,
            base_url=config.api_url or None,
            api_key_env=config.api_key_env,
        )
        agent = make_bash_agent(provider, cwd=cwd or workspace)

    # Run the agent
    start_time = time.time()
    status = "error"
    content = ""
    error = ""
    all_events = []

    try:
        state, events = agent.run(prompt, max_turns=config.max_turns)

        # Consume all events
        for event in events:
            all_events.append(event)

        elapsed = time.time() - start_time
        status = "success"

        # Extract final content from state
        for event in reversed(state.events):
            if hasattr(event, "kind") and event.kind == "message":
                msg = event.message
                if hasattr(msg, "kind") and msg.kind == "final":
                    for block in msg.content:
                        if hasattr(block, "text"):
                            content += block.text
                    break

        # Convert events to ClawEvalkit transcript format
        transcript = sal_events_to_claweval_transcript(all_events)

        # Extract usage from state
        usage = {}
        for event in state.events:
            if hasattr(event, "kind") and event.kind == "model_response":
                if hasattr(event, "usage") and event.usage:
                    u = event.usage
                    usage = {
                        "input_tokens": getattr(u, "input_tokens", 0),
                        "output_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }

        return {
            "status": status,
            "content": content,
            "transcript": transcript,
            "usage": usage,
            "execution_time": round(elapsed, 2),
            "error": error,
            "raw_events_count": len(all_events),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "content": content,
            "transcript": sal_events_to_claweval_transcript(all_events),
            "usage": {},
            "execution_time": round(elapsed, 2),
            "error": f"{type(e).__name__}: {e}",
            "raw_events_count": len(all_events),
        }


def run_pinchbench_task(task: dict, config: ModelConfig, results_dir: Path) -> dict:
    """Run a single PinchBench task using SAL's Bash Agent.

    PinchBench tasks have: id, prompt, grade_code, timeout, workspace_files.
    The agent runs in an isolated workspace, then grade_code is executed.

    Args:
        task: Task dict from PinchBench _load_tasks()
        config: Model configuration
        results_dir: Directory to save results

    Returns:
        Result dict with task_id, status, scores, mean
    """
    tid = task["id"]
    prompt = task["prompt"]
    grade_code = task.get("grade_code", "")
    workspace_files = task.get("workspace_files", [])
    timeout = task.get("timeout", 120)

    # Create isolated workspace
    workspace = Path(tempfile.mkdtemp(prefix=f"sal_pinch_{tid}_"))
    result = {
        "task_id": tid,
        "model_key": config.name,
        "status": "error",
        "scores": {},
        "mean": 0.0,
        "error": None,
    }

    try:
        # Copy workspace files
        for file_spec in workspace_files:
            if isinstance(file_spec, dict):
                if "content" in file_spec:
                    dest = workspace / file_spec["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(file_spec["content"])

        # Run agent
        agent_result = run_task_with_sal_agent(
            prompt=prompt,
            config=config,
            cwd=str(workspace),
            workspace=str(workspace),
        )

        result["status"] = agent_result["status"]
        result["execution_time"] = agent_result["execution_time"]
        result["raw_events_count"] = agent_result.get("raw_events_count", 0)

        if agent_result["error"]:
            result["error"] = agent_result["error"][:500]

        transcript = agent_result.get("transcript", [])

        # Run grading
        if grade_code:
            scores = _run_grade(grade_code, transcript, str(workspace))
            mean_score = sum(scores.values()) / len(scores) if scores else 0
            result["scores"] = scores
            result["mean"] = round(mean_score, 4)
        else:
            result["mean"] = 1.0 if agent_result["status"] == "success" else 0.0

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    # Save result + transcript
    task_output_dir = results_dir / config.name / tid
    task_output_dir.mkdir(parents=True, exist_ok=True)
    (task_output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    transcript = agent_result.get("transcript", [])
    if transcript:
        (task_output_dir / "transcript.json").write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return result


def run_agentbench_task(
    task: dict,
    config: ModelConfig,
    results_dir: Path,
    clawevalkit_dir: Path | None = None,
) -> dict:
    """Run a single AgentBench task using SAL's Bash Agent.

    AgentBench tasks are defined in task.yaml files with:
      user_message, input_files, expected_outputs, expected_behavior.

    The agent runs in an isolated workspace with input files copied in.

    Args:
        task: Task dict {"task_id", "category", "yaml_path"}
        config: Model configuration
        results_dir: Directory to save results
        clawevalkit_dir: Path to ClawEvalkit for loading task YAMLs

    Returns:
        Result dict with task_id, status, scores
    """
    tid = task["task_id"]
    yaml_path = task["yaml_path"]

    result = {
        "task_id": tid,
        "model_key": config.name,
        "status": "error",
        "scores": {},
        "error": None,
    }
    agent_result = {}

    try:
        import yaml

        if not Path(yaml_path).exists():
            result["error"] = f"YAML not found: {yaml_path}"
            return result

        cfg = yaml.safe_load(Path(yaml_path).read_text())
        user_msg = cfg.get("user_message", "")
        workspace = Path(tempfile.mkdtemp(prefix=f"sal_agentbench_{tid}_"))

        # Copy input files
        task_dir = Path(yaml_path).parent
        inputs_dir = task_dir / "inputs"
        for inp in cfg.get("input_files", []):
            fname = inp["name"] if isinstance(inp, dict) else inp
            src = inputs_dir / fname if inputs_dir.exists() else task_dir / fname
            if src.exists():
                dst = workspace / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # Run agent
        agent_result = run_task_with_sal_agent(
            prompt=user_msg,
            config=config,
            cwd=str(workspace),
            workspace=str(workspace),
        )

        result["status"] = agent_result["status"]
        result["execution_time"] = agent_result["execution_time"]
        if agent_result["error"]:
            result["error"] = agent_result["error"][:500]

        # L0 scoring: check expected outputs (file existence)
        expected_outputs = cfg.get("expected_outputs", [])
        if expected_outputs:
            l0_total = 0
            l0_passed = 0
            for eo in expected_outputs:
                pattern = eo.get("pattern", "")
                if pattern and (workspace / pattern).exists():
                    l0_passed += 1
                l0_total += 1
            l0_score = (l0_passed / l0_total * 100) if l0_total > 0 else 50
        else:
            l0_score = 50

        result["scores"] = {
            "l0_score": round(l0_score, 1),
            "l1_score": 50,
            "l2_score": 50,
            "l3_score": 50,
            "overall_score": round(
                l0_score * 0.15 + 50 * 0.25 + 50 * 0.20 + 50 * 0.25, 1
            ),
        }

        shutil.rmtree(workspace, ignore_errors=True)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    # Save result + transcript
    task_output_dir = results_dir / config.name / tid
    task_output_dir.mkdir(parents=True, exist_ok=True)
    (task_output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ab_transcript = agent_result.get("transcript", [])
    if ab_transcript:
        (task_output_dir / "transcript.json").write_text(
            json.dumps(ab_transcript, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return result


def run_clawbench_task(task: dict, config: ModelConfig, results_dir: Path) -> dict:
    """Run a single ClawBench Official task with SAL's Bash Agent.

    ClawBench tasks are directory-based:
      task.toml, instruction.md, environment/data, environment/setup.sh,
      verifier/test_output.py.

    This runner prepares an isolated workspace, asks SAL's Bash Agent to create
    the requested artifacts there, then scores the workspace with ClawBench's
    pytest verifier.
    """
    tid = task["id"]
    task_dir = Path(task["task_dir"])

    result = {
        "task_id": tid,
        "dir_name": task.get("dir_name"),
        "domain": task.get("domain"),
        "model_key": config.name,
        "status": "error",
        "passed": False,
        "score": 0.0,
        "checks_total": 0,
        "checks_passed": 0,
        "details": "",
        "error": None,
    }
    agent_result = {}
    workspace = Path(tempfile.mkdtemp(prefix=f"sal_clawbench_{tid}_"))

    try:
        _prepare_clawbench_workspace(task_dir, workspace)

        instruction = _load_clawbench_instruction(task_dir, task, workspace)
        agent_result = run_task_with_sal_agent(
            prompt=instruction,
            config=config,
            cwd=str(workspace),
            workspace=str(workspace),
        )

        result["status"] = agent_result["status"]
        result["execution_time"] = agent_result.get("execution_time", 0)
        result["raw_events_count"] = agent_result.get("raw_events_count", 0)
        if agent_result.get("error"):
            result["error"] = agent_result["error"][:500]

        verify_result = _verify_clawbench_task(task_dir, workspace)
        result["passed"] = verify_result["passed"]
        result["score"] = verify_result["score"]
        result["details"] = verify_result["details"]
        result["checks_total"] = verify_result["checks_total"]
        result["checks_passed"] = verify_result["checks_passed"]
        result["workspace_files"] = [
            str(p.relative_to(workspace))
            for p in sorted(workspace.rglob("*"))
            if p.is_file()
        ]

        if result["status"] == "success" and not result["error"]:
            result["error"] = verify_result.get("error")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        task_output_dir = results_dir / config.name / tid
        task_output_dir.mkdir(parents=True, exist_ok=True)
        (task_output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        transcript = agent_result.get("transcript", [])
        if transcript:
            (task_output_dir / "transcript.json").write_text(
                json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        shutil.rmtree(workspace, ignore_errors=True)

    return result


def _prepare_clawbench_workspace(task_dir: Path, workspace: Path) -> None:
    data_dir = task_dir / "environment" / "data"
    if data_dir.exists():
        for src in data_dir.iterdir():
            dest = workspace / src.name
            if src.is_file():
                shutil.copy2(src, dest)
            elif src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)

    setup_sh = task_dir / "environment" / "setup.sh"
    if setup_sh.exists():
        subprocess.run(
            ["bash", str(setup_sh), str(workspace.resolve())],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )


def _load_clawbench_instruction(task_dir: Path, task: dict, workspace: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        instruction = instruction_path.read_text(encoding="utf-8")
    else:
        instruction = task.get("description", "")

    abs_workspace = str(workspace.resolve())
    instruction = instruction.replace("workspace/", f"{abs_workspace}/")
    instruction = instruction.replace("`workspace/", f"`{abs_workspace}/")

    return (
        "IMPORTANT: Write every required output file under this absolute "
        f"workspace path: {abs_workspace}/\n"
        "Do not write outputs into the repository or any other directory.\n"
        "Use bash commands when useful, and create valid files that match the "
        "requested schema exactly.\n\n"
        f"{instruction}"
    )


def _verify_clawbench_task(task_dir: Path, workspace: Path) -> dict:
    bench_dir = task_dir.parents[2]
    src_dir = bench_dir / "src"

    import sys

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from claw_bench.core.verifier import verify_task

    verification = verify_task(task_dir, workspace)
    score = verification.weighted_score
    if score is None:
        score = verification.checks_passed / max(verification.checks_total, 1)

    return {
        "passed": verification.passed,
        "score": round(score, 4),
        "details": verification.details,
        "checks_total": verification.checks_total,
        "checks_passed": verification.checks_passed,
        "error": None,
    }


def _run_grade(grade_code: str, transcript: list, workspace_path: str) -> dict:
    """Execute a grade() function extracted from task markdown.

    The grade_code defines a grade(transcript, workspace_path) → dict function.
    """
    namespace = {}
    try:
        exec(grade_code, namespace)
        if "grade" in namespace:
            return namespace["grade"](transcript, workspace_path)
    except Exception:
        pass
    return {}
