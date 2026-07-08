"""Repo-chain SWE-bench Pro experiment helpers.

This module holds the pure planning/configuration pieces for the SWE-bench Pro
compression experiment where one repository maps to one long agent chain.
The executable runner builds on these helpers so tests can lock down the
research contract without starting Docker or calling a model.
"""

from __future__ import annotations

import math
import shlex
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.evals.chain import append_chain_task, start_chain_state
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import ImageBlock, MessageKind, TextBlock
from simple_agent_lab.state import State
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.tools.bash import (
    DEFAULT_BASH_MAX_ATTACH_BYTES,
    DEFAULT_BASH_MAX_OUTPUT_CHARS,
    DEFAULT_BASH_TIMEOUT_SECONDS,
    MAX_BASH_TIMEOUT_SECONDS,
    BashExecution,
    bash_execution_to_tool_result,
    budget_text,
    detect_blocked_sleep_pattern,
    interpret_command_result,
    make_bash_tool,
    strip_empty_lines,
)
from simple_agent_lab.evals.suites.swebench.patch import (
    IGNORE_BLOCK_END,
    IGNORE_BLOCK_START,
    gitignore_rules,
)

DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = ""
DEFAULT_API_KIND = "openai-responses"
DEFAULT_REASONING_EFFORT = ""
# The true model context window. Both context-management arms leave headroom
# below this and trigger at DEFAULT_THRESHOLD_TOKENS (80%) so a window's own
# work fits before the real limit.
DEFAULT_CONTEXT_WINDOW_TOKENS = 272_000
# Shared trigger point for BOTH arms: `summarize` compresses and `handoff`
# resets the window once the active context reaches this many tokens. Keeping
# them equal is what makes the handoff-vs-compression comparison fair.
DEFAULT_THRESHOLD_TOKENS = int(DEFAULT_CONTEXT_WINDOW_TOKENS * 0.8)
DEFAULT_KEEP_RECENT = 4
DEFAULT_BASELINE_TIMEOUT_SECONDS = 300
DEFAULT_PRESERVE_KINDS: tuple[MessageKind, ...] = (
    "task",
    "system",
    "context",
)
DEFAULT_MODEL_NAME = "simple-agent-lab-pro-repo-chain-bash-none"

_CONTAINER_ATTACH_MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def start_repo_state(repo: str, *, agent_name: str) -> State:
    """Create the one persistent transcript for a SWE-bench repo chain."""

    task = (
        f"Repo chain for {repo}. Solve instances for this "
        "repository in commit-time order. Carry useful context across tasks, "
        "but each instance's patch must address only the current problem."
    )
    return start_chain_state(
        task,
        agent_name=agent_name,
        metadata={"repo": repo, "chain_id": repo},
    )


def append_instance_task(
    state: State,
    *,
    agent_name: str,
    instance_id: str,
    task: str,
) -> None:
    """Append one instance prompt to an existing repo chain."""

    append_chain_task(
        state,
        agent_name=agent_name,
        item_id=instance_id,
        task=task,
        details={"swebench": {"instance_id": instance_id}},
    )


def group_instances_by_repo(
    instances: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group SWE-bench Pro rows by their ``repo`` field, preserving row order."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        repo = str(row.get("repo") or "").strip()
        if not repo:
            repo = "unknown"
        groups[repo].append(dict(row))
    return dict(groups)


def sort_repo_instances(
    instances: Sequence[Mapping[str, Any]],
    *,
    commit_times: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Sort one repo's instances by base-commit timestamp.

    Rows whose commit timestamp could not be resolved run after known commits,
    still in their original dataset order. That makes the fallback explicit and
    deterministic instead of silently pretending dataset order is commit order.
    """

    indexed = list(enumerate(instances))

    def key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int]:
        original_index, row = item
        commit = str(row.get("base_commit") or "")
        timestamp = commit_times.get(commit)
        if timestamp is None:
            return (1, original_index, 0)
        return (0, timestamp, original_index)

    return [dict(row) for _, row in sorted(indexed, key=key)]


@dataclass(frozen=True)
class RepoChainPart:
    """One contiguous, commit-ordered part of a repository chain."""

    chain_id: str
    repo: str
    part_index: int
    part_count: int
    rows: tuple[dict[str, Any], ...]


def repo_chain_part_count(instance_count: int) -> int:
    """Return how many parallel chain parts a repo should be split into."""

    if instance_count > 80:
        return 3
    if instance_count > 50:
        return 2
    return 1


def split_repo_chain_parts(
    repo: str, rows: Sequence[Mapping[str, Any]]
) -> list[RepoChainPart]:
    """Split a commit-ordered repo into contiguous, near-even chain parts."""

    copied = tuple(dict(row) for row in rows)
    part_count = repo_chain_part_count(len(copied))
    if part_count == 1:
        return [
            RepoChainPart(
                chain_id=repo,
                repo=repo,
                part_index=1,
                part_count=1,
                rows=copied,
            )
        ]

    base_size, extra = divmod(len(copied), part_count)
    parts: list[RepoChainPart] = []
    start = 0
    for offset in range(part_count):
        size = base_size + (1 if offset < extra else 0)
        end = start + size
        part_index = offset + 1
        parts.append(
            RepoChainPart(
                chain_id=f"{repo}#part-{part_index}-of-{part_count}",
                repo=repo,
                part_index=part_index,
                part_count=part_count,
                rows=copied[start:end],
            )
        )
        start = end
    return parts


class CommitTimeResolver:
    """Resolve git commit timestamps for SWE-bench Pro ``base_commit`` values."""

    def __init__(
        self,
        *,
        cache_root: Path,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.cache_root = Path(cache_root)
        self._run = run
        self._timestamps: dict[tuple[str, str], int | None] = {}
        self.warnings: list[str] = []

    def timestamp(self, repo: str, commit: str) -> int | None:
        """Return Unix commit time, or ``None`` if it cannot be resolved."""

        key = (repo, commit)
        if key in self._timestamps:
            return self._timestamps[key]
        try:
            timestamp = self._timestamp(repo, commit)
        except Exception as exc:
            timestamp = None
            self.warnings.append(
                f"{repo}@{commit}: could not resolve commit timestamp ({exc})"
            )
        self._timestamps[key] = timestamp
        return timestamp

    def timestamps(self, repo: str, commits: Iterable[str]) -> dict[str, int | None]:
        """Resolve many commit timestamps for one repo with one shallow fetch."""

        pending: list[str] = []
        resolved: dict[str, int | None] = {}
        for commit in dict.fromkeys(str(commit) for commit in commits if commit):
            key = (repo, commit)
            if key in self._timestamps:
                resolved[commit] = self._timestamps[key]
            else:
                pending.append(commit)
        if not pending:
            return resolved

        try:
            repo_dir = self._ensure_repo(repo)
            missing = [
                commit for commit in pending if not self._has_commit(repo_dir, commit)
            ]
            if missing:
                self._fetch_commits(repo_dir, missing)
            for commit in pending:
                try:
                    timestamp = self._show_timestamp(repo_dir, commit)
                except Exception as exc:
                    timestamp = None
                    self.warnings.append(
                        f"{repo}@{commit}: could not resolve commit timestamp ({exc})"
                    )
                self._timestamps[(repo, commit)] = timestamp
                resolved[commit] = timestamp
        except Exception:
            # Batch fetch can fail if one object is hidden or the local cache is
            # stale. Fall back to per-commit resolution so one bad row does not
            # discard the rest of the repo's ordering signal.
            for commit in pending:
                resolved[commit] = self.timestamp(repo, commit)
        return resolved

    def _timestamp(self, repo: str, commit: str) -> int:
        repo_dir = self._ensure_repo(repo)
        if not self._has_commit(repo_dir, commit):
            self._fetch_commits(repo_dir, [commit])
        return self._show_timestamp(repo_dir, commit)

    def _ensure_repo(self, repo: str) -> Path:
        repo_dir = self.cache_root / _safe_repo_dir(repo)
        if (repo_dir / ".git").exists():
            return repo_dir
        if repo_dir.exists():
            raise RuntimeError(f"{repo_dir} exists but is not a git repository")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        repo_dir.mkdir(parents=True, exist_ok=True)
        init = self._run(
            ["git", "-C", str(repo_dir), "init", "-q"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if int(getattr(init, "returncode", 0) or 0) != 0:
            raise RuntimeError(str(getattr(init, "stderr", "") or "git init failed"))
        remote = self._run(
            [
                "git",
                "-C",
                str(repo_dir),
                "remote",
                "add",
                "origin",
                f"https://github.com/{repo}.git",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if int(getattr(remote, "returncode", 0) or 0) != 0:
            raise RuntimeError(
                str(getattr(remote, "stderr", "") or "git remote add failed")
            )
        return repo_dir

    def _has_commit(self, repo_dir: Path, commit: str) -> bool:
        result = self._run(
            ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return int(getattr(result, "returncode", 0) or 0) == 0

    def _fetch_commits(self, repo_dir: Path, commits: Sequence[str]) -> None:
        if not commits:
            return
        fetch = self._run(
            [
                "git",
                "-C",
                str(repo_dir),
                "-c",
                "protocol.version=2",
                "fetch",
                "--filter=blob:none",
                "--no-tags",
                "--depth=1",
                "origin",
                *commits,
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if int(getattr(fetch, "returncode", 0) or 0) != 0:
            raise RuntimeError(str(getattr(fetch, "stderr", "") or "git fetch failed"))

    def _show_timestamp(self, repo_dir: Path, commit: str) -> int:
        show = self._run(
            ["git", "-C", str(repo_dir), "show", "-s", "--format=%ct", commit],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if int(getattr(show, "returncode", 0) or 0) != 0:
            raise RuntimeError(str(getattr(show, "stderr", "") or "git show failed"))
        return int(str(getattr(show, "stdout", "")).strip())


def _safe_repo_dir(repo: str) -> str:
    return repo.strip().replace("/", "__").replace("\\", "__")


@dataclass(frozen=True)
class ProRepoExperimentConfig:
    """Operator-visible configuration for the Pro repo-chain experiment."""

    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    model: str = DEFAULT_MODEL
    api_kind: str = DEFAULT_API_KIND
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_turns: int = 250
    # Handoff trigger. Defaults to the same value as ``threshold_tokens`` (the
    # summarize trigger) so handoff and compression reset the window at the same
    # point for a fair comparison; override ``--context-window-tokens`` to study
    # a different handoff trigger in isolation.
    context_window_tokens: int = DEFAULT_THRESHOLD_TOKENS
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS
    keep_recent: int = DEFAULT_KEEP_RECENT
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS
    agent_flavor: str = "bash"
    solver_read: bool = False
    task_tool: bool = False
    compression_strategy: str = "none"
    handoff: bool = True
    model_name: str = DEFAULT_MODEL_NAME

    def context_policy(
        self,
        provider: Provider,
        *,
        request_extra: Mapping[str, Any] | None = None,
    ) -> ContextPolicy:
        """Build the configured LLM summarization policy using ``provider``."""

        if self.compression_strategy != "summarize":
            return ContextPolicy()
        compressor = make_llm_agent(
            name="swebench_compressor",
            provider=provider,
            role=(
                "Summarize older repo-chain context. Preserve durable "
                "facts, decisions, tool results, constraints, file paths, test "
                "signals, and unresolved questions. Omit low-value wording."
            ),
            request_extra=request_extra,
        )
        return ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=compressor,
                threshold_tokens=self.threshold_tokens,
                keep_recent=self.keep_recent,
                preserve_kinds=self.preserve_kinds,
            )
        )

    def as_record(self) -> dict[str, Any]:
        """JSON-friendly experiment manifest fragment."""

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "model": self.model,
            "api_kind": self.api_kind,
            "reasoning_effort": self.reasoning_effort,
            "max_turns": self.max_turns,
            "context_window_tokens": self.context_window_tokens,
            "threshold_tokens": self.threshold_tokens,
            "keep_recent": self.keep_recent,
            "preserve_kinds": list(self.preserve_kinds),
            "agent_flavor": self.agent_flavor,
            "solver_read": self.solver_read,
            "task_tool": self.task_tool,
            "compression_strategy": self.compression_strategy,
            "handoff": self.handoff,
            "model_name": self.model_name,
        }


@dataclass
class CurrentContainer:
    """Mutable pointer to the instance container the host-side bash tool targets."""

    name: str = ""
    workdir: str = "/app"


class DockerCommandRunner:
    """Small subprocess wrapper for Docker CLI commands used by repo chains."""

    def __init__(
        self,
        *,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self._run = run

    def start_container(
        self,
        *,
        container_name: str,
        image: str,
        workdir: str,
        network_mode: str = "host",
        mem_limit: str | None = "8g",
    ) -> Any:
        """Start a long-lived SWE-bench Pro instance container."""

        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            "",
        ]
        if network_mode:
            command.extend(["--network", network_mode])
        if mem_limit:
            command.extend(["--memory", mem_limit])
        command.extend(
            [
                "-w",
                workdir,
                image,
                "/bin/sh",
                "-lc",
                "sleep infinity",
            ]
        )
        return self._run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )

    def exec(
        self,
        *,
        container_name: str,
        workdir: str,
        command: str,
        timeout_seconds: float,
    ) -> Any:
        """Run one shell command inside ``container_name``."""

        return self._run(
            [
                "docker",
                "exec",
                "-w",
                workdir,
                container_name,
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )

    def remove_container(self, container_name: str) -> Any:
        """Force-remove a container if it exists."""

        return self._run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )


def make_container_bash_tool(
    current: CurrentContainer,
    docker: DockerCommandRunner,
    *,
    default_timeout_seconds: float = DEFAULT_BASH_TIMEOUT_SECONDS,
    max_timeout_seconds: float = MAX_BASH_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_BASH_MAX_OUTPUT_CHARS,
    max_attach_bytes: int = DEFAULT_BASH_MAX_ATTACH_BYTES,
) -> AgentTool:
    """Return a bash tool that executes inside the current SWE-bench container."""

    def execute(call_id, args, abort, on_update):
        del call_id, on_update
        if abort():
            return text_result("Bash command aborted before start.", is_error=True)
        if not current.name:
            return text_result(
                "No active repository container is attached to this repo chain.",
                is_error=True,
            )
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
            timeout_seconds = _resolve_timeout_seconds(
                args.get("timeout_seconds"),
                default_timeout_seconds=default_timeout_seconds,
                max_timeout_seconds=max_timeout_seconds,
            )
        except ValueError as exc:
            return text_result(f"Invalid bash timeout: {exc}", is_error=True)

        started = time.monotonic()
        try:
            completed = docker.exec(
                container_name=current.name,
                workdir=current.workdir,
                command=command,
                timeout_seconds=timeout_seconds,
            )
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
            exit_code = int(getattr(completed, "returncode", 0) or 0)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_process_text(exc.stdout)
            stderr = _coerce_process_text(exc.stderr)
            stderr = (
                f"{stderr}\nTimed out after {timeout_seconds:g}s"
                if stderr
                else f"Timed out after {timeout_seconds:g}s"
            )
            exit_code = -1
            timed_out = True
        except OSError as exc:
            stdout = ""
            stderr = f"{type(exc).__name__}: {exc}"
            exit_code = -1
            timed_out = False

        clean_stdout = strip_empty_lines(stdout)
        clean_stderr = strip_empty_lines(stderr)
        visible_stdout, stdout_truncated = budget_text(clean_stdout, max_output_chars)
        visible_stderr, stderr_truncated = budget_text(clean_stderr, max_output_chars)
        interpretation = interpret_command_result(command, exit_code)
        execution = BashExecution(
            command=command,
            cwd=f"docker:{current.name}:{current.workdir}",
            exit_code=exit_code,
            stdout=visible_stdout,
            stderr=visible_stderr,
            elapsed_seconds=time.monotonic() - started,
            raw_stdout=clean_stdout,
            raw_stderr=clean_stderr,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            interpretation=interpretation.message,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            is_error=timed_out or exit_code < 0 or interpretation.is_error,
        )
        result = bash_execution_to_tool_result(execution)
        attach = args.get("attach")
        if attach:
            result = _attach_container_files(
                result,
                paths=attach,
                current=current,
                docker=docker,
                max_attach_bytes=max_attach_bytes,
            )
        return result

    standard = make_bash_tool(
        default_timeout_seconds=default_timeout_seconds,
        max_timeout_seconds=max_timeout_seconds,
        max_output_chars=max_output_chars,
        max_attach_bytes=max_attach_bytes,
    )
    return replace(standard, execute=execute)


def _attach_container_files(
    result: ToolResult,
    *,
    paths: Any,
    current: CurrentContainer,
    docker: DockerCommandRunner,
    max_attach_bytes: int,
) -> ToolResult:
    """Append image blocks for requested paths inside the active container."""

    if not isinstance(paths, (list, tuple)):
        return _attach_container_error(
            result, f"`attach` must be a list of paths, got {type(paths).__name__}"
        )

    extras: list[ImageBlock] = []
    notes: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            notes.append(f"skipped attach entry {raw!r}: not a non-empty string")
            continue
        suffix = Path(raw).suffix.lower()
        mime = _CONTAINER_ATTACH_MIME_BY_SUFFIX.get(suffix)
        if mime is None:
            notes.append(
                f"attach path {raw!r}: unsupported extension {suffix!r}"
                f" (allowed: {sorted(_CONTAINER_ATTACH_MIME_BY_SUFFIX)})"
            )
            continue
        probe = docker.exec(
            container_name=current.name,
            workdir=current.workdir,
            command=_container_attach_command(raw, max_attach_bytes),
            timeout_seconds=DEFAULT_BASH_TIMEOUT_SECONDS,
        )
        returncode = int(getattr(probe, "returncode", 0) or 0)
        stdout = strip_empty_lines(str(getattr(probe, "stdout", "") or ""))
        stderr = strip_empty_lines(str(getattr(probe, "stderr", "") or ""))
        if returncode == 0:
            extras.append(ImageBlock(data="".join(stdout.split()), mime_type=mime))
            continue
        if stdout.startswith("__SAL_ATTACH_NOT_FILE__"):
            notes.append(f"attach path {raw!r} is not a file")
            continue
        if stdout.startswith("__SAL_ATTACH_TOO_LARGE__:"):
            size = stdout.partition(":")[2] or "unknown"
            notes.append(
                f"attach path {raw!r}: {size} bytes exceeds limit "
                f"{max_attach_bytes} bytes"
            )
            continue
        details = stderr or stdout or f"docker exec exited {returncode}"
        notes.append(f"failed to read attach path {raw!r}: {details}")

    if not extras and not notes:
        return result

    new_content: list[TextBlock | ImageBlock] = list(result.content)
    new_content.extend(extras)
    if notes:
        note_text = "attach notes:\n" + "\n".join(f"- {note}" for note in notes)
        new_content.append(TextBlock(note_text))
    new_details = (
        dict(result.details)
        if isinstance(result.details, dict)
        else {"details": result.details}
    )
    if notes:
        new_details["attach_notes"] = list(notes)
    all_failed = bool(notes) and not extras
    return replace(
        result,
        content=tuple(new_content),
        details=new_details,
        is_error=result.is_error or all_failed,
    )


def _attach_container_error(result: ToolResult, message: str) -> ToolResult:
    new_content = (*result.content, TextBlock(text=f"attach error: {message}"))
    return replace(result, content=new_content, is_error=True)


def _container_attach_command(path: str, max_attach_bytes: int) -> str:
    quoted = shlex.quote(path)
    return (
        f"path={quoted}\n"
        'if [ ! -f "$path" ]; then printf "%s\\n" "__SAL_ATTACH_NOT_FILE__"; exit 3; fi\n'
        'size=$(wc -c < "$path" | tr -d "[:space:]")\n'
        f'if [ "${{size:-0}}" -gt {max_attach_bytes} ]; then '
        'printf "%s:%s\\n" "__SAL_ATTACH_TOO_LARGE__" "$size"; exit 4; fi\n'
        'base64 < "$path" | tr -d "\\n"\n'
    )


def prepare_container_baseline(
    docker: DockerCommandRunner,
    current: CurrentContainer,
    *,
    language: str,
) -> str:
    """Commit the container's pre-agent state and return the baseline commit."""

    _install_container_exclude(docker, current, language=language)
    result = docker.exec(
        container_name=current.name,
        workdir=current.workdir,
        command=(
            "set -e\n"
            "git add -A\n"
            "if ! git diff --cached --quiet; then\n"
            "  git -c user.name='Simple Agent Lab' "
            "-c user.email='simple-agent-lab@example.invalid' "
            "commit --no-verify -m 'simple-agent-lab pre-agent baseline' "
            ">/dev/null\n"
            "fi\n"
            "git rev-parse HEAD\n"
        ),
        timeout_seconds=DEFAULT_BASELINE_TIMEOUT_SECONDS,
    )
    _raise_on_failure(result, "prepare baseline")
    return str(getattr(result, "stdout", "") or "").strip().splitlines()[-1]


def extract_container_patch(
    docker: DockerCommandRunner,
    current: CurrentContainer,
    *,
    language: str,
    baseline_commit: str,
) -> str:
    """Return a SWE-bench-style git diff from the current container."""

    _install_container_exclude(docker, current, language=language)
    diff_command = "git add -A\n"
    diff_command += "git diff --cached --src-prefix=a/ --dst-prefix=b/"
    if baseline_commit:
        diff_command += f" {baseline_commit}"
    result = docker.exec(
        container_name=current.name,
        workdir=current.workdir,
        command=diff_command,
        timeout_seconds=60,
    )
    _raise_on_failure(result, "extract patch")
    patch = str(getattr(result, "stdout", "") or "").strip()
    return patch + ("\n" if patch else "")


def _install_container_exclude(
    docker: DockerCommandRunner,
    current: CurrentContainer,
    *,
    language: str,
) -> None:
    block = "\n".join(
        [
            IGNORE_BLOCK_START,
            *gitignore_rules(language),
            IGNORE_BLOCK_END,
        ]
    )
    result = docker.exec(
        container_name=current.name,
        workdir=current.workdir,
        command=f"mkdir -p .git/info && cat > .git/info/exclude <<'EOF'\n{block}\nEOF\n",
        timeout_seconds=30,
    )
    _raise_on_failure(result, "install generated-file exclude rules")


def _raise_on_failure(result: Any, label: str) -> None:
    returncode = int(getattr(result, "returncode", 0) or 0)
    if returncode == 0:
        return
    stderr = str(getattr(result, "stderr", "") or "").strip()
    raise RuntimeError(f"{label} failed with exit {returncode}: {stderr}")


def _resolve_timeout_seconds(
    value: Any,
    *,
    default_timeout_seconds: float,
    max_timeout_seconds: float,
) -> float:
    if value is None or value == "":
        return default_timeout_seconds
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"timeout_seconds must be numeric, got {value!r}") from None
    if math.isnan(timeout):
        raise ValueError("timeout_seconds must be a real number, got NaN")
    if timeout <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {value!r}")
    return min(timeout, max_timeout_seconds)


def _coerce_process_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
