"""Python bridge that turns JS ``agent()`` calls into Agent.run calls."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import AssistantMessage
from simple_agent_lab.state import State
from simple_agent_lab.tools import AgentTool
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool
from simple_agent_lab.trace import run_trace_from_state, trace_record
from simple_agent_lab.workflow.base import final_output


@dataclass(frozen=True)
class AgentCallOptions:
    """Options accepted by the JavaScript ``agent(prompt, opts)`` primitive."""

    name: str = "agent"
    role: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: tuple[str, ...] = ()
    max_turns: int = 10
    timeout_seconds: float | None = None
    worktree: bool | str = False
    schema: Mapping[str, Any] | None = None
    cache_key: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AgentCallOptions":
        data = dict(raw or {})
        tools_raw = data.get("tools") or ()
        if isinstance(tools_raw, str):
            tools = (tools_raw,)
        else:
            tools = tuple(str(item) for item in tools_raw)
        max_turns = data.get("maxTurns", data.get("max_turns", 10))
        timeout = data.get("timeoutSeconds", data.get("timeout_seconds"))
        schema = data.get("schema")
        return cls(
            name=str(data.get("name") or "agent"),
            role=str(data.get("role") or ""),
            system_prompt=str(
                data.get("systemPrompt") or data.get("system_prompt") or ""
            ),
            model=str(data.get("model") or ""),
            tools=tools,
            max_turns=max(1, int(max_turns or 10)),
            timeout_seconds=(float(timeout) if timeout is not None else None),
            worktree=cast("bool | str", data.get("worktree") or False),
            schema=cast("Mapping[str, Any] | None", schema)
            if isinstance(schema, Mapping)
            else None,
            cache_key=str(data.get("cacheKey") or data.get("cache_key") or ""),
        )


@dataclass(frozen=True)
class AgentCallResult:
    """Serializable result returned to the workflow script."""

    call_id: str
    name: str
    phase: str
    status: str
    output: str
    trace_path: str
    model: str = ""
    usage: dict[str, int] | None = None
    structured: Any = None
    schema_error: str = ""
    reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "phase": self.phase,
            "status": self.status,
            "output": self.output,
            "trace_path": self.trace_path,
            "model": self.model,
            "usage": self.usage or {},
            "structured": self.structured,
            "schema_error": self.schema_error,
            "reused": self.reused,
        }


class AgentCallRunner(Protocol):
    def run_agent(
        self,
        prompt: str,
        *,
        options: AgentCallOptions,
        call_id: str,
        phase: str,
        artifacts_dir: Path,
    ) -> AgentCallResult: ...


AgentBuilder = Callable[[AgentCallOptions, Provider, Path], Agent]
ToolFactory = Callable[[Path], AgentTool]


class SimpleAgentCallRunner:
    """Default bridge runner backed by ``make_llm_agent`` and ``Agent.run``."""

    def __init__(
        self,
        *,
        provider: Provider,
        cwd: str | Path,
        request_extra: Mapping[str, Any] | None = None,
        default_system_prompt: str = "",
        default_role: str = "",
        default_tools: Sequence[str] = (),
        tool_factories: Mapping[str, ToolFactory] | None = None,
        build_agent: AgentBuilder | None = None,
        allow_worktrees: bool = True,
        max_turns_cap: int | None = None,
    ) -> None:
        self.provider = provider
        self.cwd = Path(cwd)
        self.request_extra = dict(request_extra or {})
        self.default_system_prompt = default_system_prompt
        self.default_role = default_role
        self.default_tools = tuple(default_tools)
        self.tool_factories = dict(tool_factories or _default_tool_factories())
        self.build_agent = build_agent or self._default_build_agent
        self.allow_worktrees = allow_worktrees
        self.max_turns_cap = (
            max(1, max_turns_cap) if max_turns_cap is not None else None
        )

    def run_agent(
        self,
        prompt: str,
        *,
        options: AgentCallOptions,
        call_id: str,
        phase: str,
        artifacts_dir: Path,
    ) -> AgentCallResult:
        if self.max_turns_cap is not None and options.max_turns > self.max_turns_cap:
            options = dataclasses.replace(options, max_turns=self.max_turns_cap)
        workdir = self._workdir_for(options, call_id, artifacts_dir)
        provider = self._provider_for(options)
        agent = self.build_agent(options, provider, workdir)
        state, events = agent.run(prompt, max_turns=options.max_turns)
        for _ in events:
            pass
        output = final_output(state, agent.name)
        trace_path = self._write_trace(
            state=state,
            call_id=call_id,
            agent_name=agent.name,
            phase=phase,
            artifacts_dir=artifacts_dir,
        )
        structured, schema_error = _structured_output(output, options.schema)
        return AgentCallResult(
            call_id=call_id,
            name=agent.name,
            phase=phase,
            status="completed",
            output=output,
            trace_path=str(trace_path),
            model=_last_model(state) or provider.model,
            usage=_usage_summary(state),
            structured=structured,
            schema_error=schema_error,
        )

    def _default_build_agent(
        self, options: AgentCallOptions, provider: Provider, cwd: Path
    ) -> Agent:
        tools = self._tools_for(options, cwd)
        return make_llm_agent(
            name=options.name,
            provider=provider,
            role=options.role or self.default_role,
            tools=tools,
            system_prompt=options.system_prompt or self.default_system_prompt,
            target="user",
            request_extra=self.request_extra,
            timeout_seconds=options.timeout_seconds,
        )

    def _provider_for(self, options: AgentCallOptions) -> Provider:
        if not options.model:
            return self.provider
        return dataclasses.replace(self.provider, model=options.model)

    def _tools_for(self, options: AgentCallOptions, cwd: Path) -> tuple[AgentTool, ...]:
        names = options.tools or self.default_tools
        tools: list[AgentTool] = []
        for name in names:
            factory = self.tool_factories.get(name)
            if factory is None:
                raise ValueError(
                    f"Unsupported dynamic workflow tool {name!r}; "
                    f"allowed: {sorted(self.tool_factories)}"
                )
            tools.append(factory(cwd))
        return tuple(tools)

    def _workdir_for(
        self, options: AgentCallOptions, call_id: str, artifacts_dir: Path
    ) -> Path:
        if not options.worktree:
            return self.cwd
        if not self.allow_worktrees:
            raise ValueError("dynamic workflow worktrees are disabled for this suite")
        label = call_id if options.worktree is True else str(options.worktree)
        target = artifacts_dir / "worktrees" / _safe_part(label)
        if target.exists():
            return target
        repo_root = _git_root(self.cwd)
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git",
            "-C",
            str(repo_root),
            "worktree",
            "add",
            "--detach",
            str(target),
            "HEAD",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"failed to create worktree {target}: {detail}")
        return target

    def _write_trace(
        self,
        *,
        state: State,
        call_id: str,
        agent_name: str,
        phase: str,
        artifacts_dir: Path,
    ) -> Path:
        trace_dir = artifacts_dir / "subagents" / _safe_part(call_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trajectory.jsonl"
        trace = run_trace_from_state(
            state=state,
            trace_id=f"dynamic.{call_id}.{agent_name}",
            producer="dynamic_workflow",
            meta={"call_id": call_id, "phase": phase, "agent": agent_name},
        )
        trace_path.write_text(
            json.dumps(trace_record(trace), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return trace_path


def _default_tool_factories() -> dict[str, ToolFactory]:
    return {
        "bash": lambda cwd: make_bash_tool(cwd=cwd),
        "read": lambda cwd: make_read_tool(cwd=cwd),
    }


def _structured_output(
    output: str, schema: Mapping[str, Any] | None
) -> tuple[Any, str]:
    if not schema:
        return None, ""
    try:
        return json.loads(output), ""
    except json.JSONDecodeError as exc:
        return None, f"output was not valid JSON for requested schema: {exc}"


def _usage_summary(state: State) -> dict[str, int]:
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
    }
    for message in state.messages:
        if not isinstance(message, AssistantMessage) or message.usage is None:
            continue
        total["input_tokens"] += message.usage.input_tokens
        total["output_tokens"] += message.usage.output_tokens
        total["cache_read_tokens"] += message.usage.cache_read_tokens
        total["cache_write_tokens"] += message.usage.cache_write_tokens
        total["total_tokens"] += message.usage.total_tokens
    return total


def _last_model(state: State) -> str:
    for message in reversed(state.messages):
        if isinstance(message, AssistantMessage) and message.model:
            return message.model
    return ""


def _git_root(cwd: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"dynamic workflow worktree requires a git repo: {detail}")
    return Path(result.stdout.strip())


def _safe_part(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    return safe or "item"
