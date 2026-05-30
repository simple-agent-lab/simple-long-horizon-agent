"""Small bash tool for local demos.

This is intentionally not a production shell sandbox. It is a compact teaching
tool that shows how a model-visible tool call becomes a local process result and
then a normal `tool_result` message in the runtime transcript.
"""

from __future__ import annotations

import base64
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from simple_agent_lab.messages import ImageBlock, TextBlock

from . import (
    AbortFlag,
    AgentTool,
    ToolExecutionMode,
    ToolResult,
    ToolUpdateFn,
    text_result,
)


BASH_TOOL_NAME = "bash"

DEFAULT_BASH_TIMEOUT_SECONDS = 30.0
DEFAULT_BASH_MAX_OUTPUT_CHARS = 4000
MAX_BASH_TIMEOUT_SECONDS = 60.0
DEFAULT_BASH_MAX_ATTACH_BYTES = 5 * 1024 * 1024  # 5 MiB per attached image

# Env vars we inject (additively) into the bash subprocess so that paging tools
# and progress bars do not blow the model-visible output budget. Mirrors the
# defaults used by mini-swe-agent's SWE-bench config. We do not override values
# the caller already exported, so the user keeps full control when they want
# colored or paged output for their own debugging.
NON_INTERACTIVE_BASH_ENV: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

_IMAGE_MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class CommandInterpretation:
    """Semantic reading of a shell exit code."""

    is_error: bool
    message: str = ""


@dataclass(frozen=True)
class BashExecution:
    """Structured result from one bash command."""

    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    raw_stdout: str = ""
    raw_stderr: str = ""
    timed_out: bool = False
    timeout_seconds: float | None = None
    interpretation: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    is_error: bool = False


def make_bash_tool(
    *,
    cwd: str | Path | None = None,
    default_timeout_seconds: float = DEFAULT_BASH_TIMEOUT_SECONDS,
    max_timeout_seconds: float = MAX_BASH_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_BASH_MAX_OUTPUT_CHARS,
    max_attach_bytes: int = DEFAULT_BASH_MAX_ATTACH_BYTES,
    execution_mode: ToolExecutionMode = "parallel",
) -> AgentTool:
    """Return an `AgentTool` that executes one local bash command."""

    if default_timeout_seconds <= 0:
        raise ValueError("default_timeout_seconds must be > 0")
    if max_timeout_seconds <= 0:
        raise ValueError("max_timeout_seconds must be > 0")
    if max_output_chars <= 0:
        raise ValueError("max_output_chars must be > 0")

    root = Path(cwd or ".").resolve()

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Bash command aborted before start.", is_error=True)

        command = str(args.get("command", "")).strip()
        if not command:
            return text_result(
                "Missing required bash argument: command.", is_error=True
            )

        blocked_sleep = detect_blocked_sleep_pattern(command)
        if blocked_sleep is not None:
            return text_result(
                f"Blocked bash command: {blocked_sleep}. Use a shorter delay or a real readiness check.",
                details={"command": command, "blocked_sleep": blocked_sleep},
                is_error=True,
            )

        try:
            timeout_seconds = _resolve_timeout(
                args.get("timeout_seconds"),
                default_timeout_seconds,
                max_timeout_seconds,
            )
        except ValueError as exc:
            return text_result(f"Invalid bash timeout: {exc}", is_error=True)
        execution = run_bash(
            command,
            cwd=root,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        if abort():
            return text_result(
                "Bash command completed, but the run was aborted before observation.",
                details=asdict(execution),
                is_error=True,
            )
        result = bash_execution_to_tool_result(execution)
        attach = args.get("attach")
        if attach:
            result = _attach_files(
                result,
                paths=attach,
                root=root,
                max_attach_bytes=max_attach_bytes,
            )
        return result

    return AgentTool(
        name=BASH_TOOL_NAME,
        description="Run a bash command in the local workspace and return stdout/stderr.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "description": {
                    "type": "string",
                    "description": "Short active-voice description of the command.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": f"Optional timeout in seconds, capped at {max_timeout_seconds:g}.",
                },
                "attach": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of file paths to inline in the tool result as image "
                        "blocks. Use this when the command produces a visual artifact you want "
                        "the model to see (screenshot, plot, rendered diagram). Supports PNG, "
                        f"JPEG, GIF, WebP up to {max_attach_bytes // (1024 * 1024)} MiB each. "
                        "Paths relative to the workspace cwd are resolved against it."
                    ),
                },
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
        timeout_seconds=max_timeout_seconds + 1,
    )


def run_bash(
    command: str,
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = DEFAULT_BASH_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_BASH_MAX_OUTPUT_CHARS,
) -> BashExecution:
    """Execute `command` with `bash -lc` and return a structured result."""

    if not command.strip():
        raise ValueError("command must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if max_output_chars <= 0:
        raise ValueError("max_output_chars must be > 0")

    root = Path(cwd or ".").resolve()
    env = _bash_subprocess_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=root,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_process_text(exc.stdout)
        stderr = _coerce_process_text(exc.stderr)
        if stderr:
            stderr = f"{stderr}\nTimed out after {timeout_seconds:g}s"
        else:
            stderr = f"Timed out after {timeout_seconds:g}s"
        exit_code = -1
        timed_out = True
    except OSError as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exit_code = -1
        timed_out = False

    elapsed = time.monotonic() - start
    clean_stdout = strip_empty_lines(stdout)
    clean_stderr = strip_empty_lines(stderr)
    visible_stdout, stdout_truncated = budget_text(clean_stdout, max_output_chars)
    visible_stderr, stderr_truncated = budget_text(clean_stderr, max_output_chars)
    interpretation = interpret_command_result(command, exit_code)
    is_error = timed_out or exit_code < 0 or interpretation.is_error
    return BashExecution(
        command=command,
        cwd=str(root),
        exit_code=exit_code,
        stdout=visible_stdout,
        stderr=visible_stderr,
        raw_stdout=clean_stdout,
        raw_stderr=clean_stderr,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        interpretation=interpretation.message,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        is_error=is_error,
    )


def bash_execution_to_tool_result(execution: BashExecution) -> ToolResult:
    """Convert a structured bash execution into a model-visible tool result."""

    return text_result(
        format_bash_observation(execution),
        details=asdict(execution),
        is_error=execution.is_error,
    )


def format_bash_observation(execution: BashExecution) -> str:
    """Render the compact text that goes back to the model."""

    lines = [f"$ {execution.command}"]
    if execution.stdout:
        lines.extend(["stdout:", execution.stdout])
    if execution.stderr:
        lines.extend(["stderr:", execution.stderr])
    if not execution.stdout and not execution.stderr:
        if not execution.is_error:
            lines.append("Done. Command completed with no output.")
        else:
            lines.append("(no output)")
    if execution.interpretation:
        lines.append(f"note: {execution.interpretation}")
    if execution.exit_code != 0:
        lines.append(f"exit_code: {execution.exit_code}")
    if execution.stdout_truncated or execution.stderr_truncated:
        lines.append(
            "note: output was truncated for model context. "
            "Re-run with a narrower view: `head`/`tail`, `sed -n 'A,Bp' FILE`, "
            "or a more selective `grep`/`find` pattern."
        )
    return "\n".join(lines)


def interpret_command_result(command: str, exit_code: int) -> CommandInterpretation:
    """Interpret exit codes for common shell commands.

    `grep`/`rg` no-match and `diff` differences are observations, not failures.
    """

    base_command = _last_base_command(command)
    if base_command in {"grep", "rg"}:
        if exit_code == 1:
            return CommandInterpretation(False, "No matches found")
        return CommandInterpretation(exit_code >= 2, _failure_message(exit_code))
    if base_command == "find":
        if exit_code == 1:
            return CommandInterpretation(False, "Some directories were inaccessible")
        return CommandInterpretation(exit_code >= 2, _failure_message(exit_code))
    if base_command == "diff":
        if exit_code == 1:
            return CommandInterpretation(False, "Files differ")
        return CommandInterpretation(exit_code >= 2, _failure_message(exit_code))
    if base_command in {"test", "["}:
        if exit_code == 1:
            return CommandInterpretation(False, "Condition is false")
        return CommandInterpretation(exit_code >= 2, _failure_message(exit_code))
    return CommandInterpretation(exit_code != 0, _failure_message(exit_code))


def detect_blocked_sleep_pattern(command: str) -> str | None:
    """Detect long leading sleeps that block demos."""

    parts = re.split(r"\s*(?:&&|;)\s*", command.strip(), maxsplit=1)
    first = parts[0] if parts else ""
    match = re.fullmatch(r"sleep\s+(\d+)", first)
    if match is None:
        return None
    seconds = int(match.group(1))
    if seconds < 2:
        return None
    rest = parts[1].strip() if len(parts) > 1 else ""
    return f"sleep {seconds} before {rest}" if rest else f"standalone sleep {seconds}"


def strip_empty_lines(content: str) -> str:
    """Strip only leading/trailing empty lines while preserving inner spacing."""

    lines = content.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines) - 1
    while end >= start and not lines[end].strip():
        end -= 1
    if start > end:
        return ""
    return "\n".join(lines[start : end + 1])


def budget_text(content: str, max_chars: int) -> tuple[str, bool]:
    """Keep model-visible output small while preserving both head and tail."""

    if len(content) <= max_chars:
        return content, False
    # Reserve marker space using `len(content)` as the upper bound on the
    # printed `omitted` count — guarantees the real marker fits in the budget.
    reserved = len(f"\n... [truncated {len(content)} chars] ...\n")
    keep = max(0, max_chars - reserved)
    omitted = len(content) - keep
    marker = f"\n... [truncated {omitted} chars] ...\n"
    head_chars = keep // 2
    tail_chars = keep - head_chars
    return f"{content[:head_chars]}{marker}{content[-tail_chars:]}", True


def _resolve_timeout(
    requested: Any,
    default_timeout_seconds: float,
    max_timeout_seconds: float,
) -> float:
    if requested is None or requested == "":
        return default_timeout_seconds
    try:
        timeout = float(requested)
    except (TypeError, ValueError):
        raise ValueError(
            f"timeout_seconds must be numeric, got {requested!r}"
        ) from None
    if math.isnan(timeout):
        raise ValueError("timeout_seconds must be a real number, got NaN")
    if timeout <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {requested!r}")
    return min(timeout, max_timeout_seconds)


def _last_base_command(command: str) -> str:
    segments = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command.strip())
    last = next((segment for segment in reversed(segments) if segment.strip()), command)
    try:
        parts = shlex.split(last)
    except ValueError:
        parts = last.strip().split()
    return parts[0] if parts else ""


def _failure_message(exit_code: int) -> str:
    return f"Command failed with exit code {exit_code}" if exit_code != 0 else ""


def _bash_subprocess_env() -> dict[str, str]:
    """Return a copy of os.environ with non-interactive defaults filled in.

    Caller-provided values always win so the user can override (e.g. set
    ``PAGER=less`` if they want paging back in a local terminal demo).
    """
    env = os.environ.copy()
    for key, value in NON_INTERACTIVE_BASH_ENV.items():
        env.setdefault(key, value)
    return env


def _coerce_process_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _attach_files(
    result: ToolResult,
    *,
    paths: Any,
    root: Path,
    max_attach_bytes: int,
) -> ToolResult:
    """Append image content blocks for each requested attachment path."""

    if not isinstance(paths, (list, tuple)):
        return _attach_error(
            result, f"`attach` must be a list of paths, got {type(paths).__name__}"
        )

    extras: list[ImageBlock] = []
    notes: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            notes.append(f"skipped attach entry {raw!r}: not a non-empty string")
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            candidate = candidate.resolve()
        except OSError as exc:
            notes.append(f"could not resolve attach path {raw!r}: {exc}")
            continue
        if not candidate.is_file():
            notes.append(f"attach path {raw!r} is not a file")
            continue
        mime = _IMAGE_MIME_BY_SUFFIX.get(candidate.suffix.lower())
        if mime is None:
            notes.append(
                f"attach path {raw!r}: unsupported extension {candidate.suffix!r}"
                f" (allowed: {sorted(_IMAGE_MIME_BY_SUFFIX)})"
            )
            continue
        size = candidate.stat().st_size
        if size > max_attach_bytes:
            notes.append(
                f"attach path {raw!r}: {size} bytes exceeds limit "
                f"{max_attach_bytes} bytes"
            )
            continue
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            notes.append(f"failed to read attach path {raw!r}: {exc}")
            continue
        extras.append(
            ImageBlock(data=base64.b64encode(data).decode("ascii"), mime_type=mime)
        )

    if not extras and not notes:
        return result

    new_content: list[TextBlock | ImageBlock] = list(result.content)
    new_content.extend(extras)
    if notes:
        note_text = "attach notes:\n" + "\n".join(f"- {n}" for n in notes)
        new_content.append(TextBlock(text=note_text))
    new_details = (
        dict(result.details)
        if isinstance(result.details, dict)
        else {"details": result.details}
    )
    if notes:
        new_details["attach_notes"] = list(notes)
    # If every requested path failed, surface that as an error.
    all_failed = bool(notes) and not extras
    return replace(
        result,
        content=tuple(new_content),
        details=new_details,
        is_error=result.is_error or all_failed,
    )


def _attach_error(result: ToolResult, message: str) -> ToolResult:
    new_content = (*result.content, TextBlock(text=f"attach error: {message}"))
    return replace(result, content=new_content, is_error=True)
