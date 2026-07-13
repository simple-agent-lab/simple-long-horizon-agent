"""Node-backed JavaScript runtime for dynamic workflows."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .bridge import AgentCallOptions, AgentCallRunner
from .journal import WorkflowJournal


@dataclass(frozen=True)
class WorkflowRuntimeOptions:
    max_concurrency: int = 16
    max_agents: int = 1000
    timeout_seconds: float | None = None
    node_binary: str = "node"
    process_prefix: tuple[str, ...] = ()
    enforce_node_permissions: bool = True


@dataclass(frozen=True)
class WorkflowRunResult:
    output: str
    raw_result: Any
    script_path: Path
    journal_path: Path
    result_path: Path
    artifacts_dir: Path
    agent_calls: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "result": self.raw_result,
            "script_path": str(self.script_path),
            "journal_path": str(self.journal_path),
            "result_path": str(self.result_path),
            "artifacts_dir": str(self.artifacts_dir),
            "agent_calls": list(self.agent_calls),
        }


class DynamicWorkflowRuntime:
    """Execute a workflow JavaScript script and service its subagent calls."""

    def __init__(
        self,
        *,
        runner: AgentCallRunner,
        options: WorkflowRuntimeOptions | None = None,
        saved_workflows: Mapping[str, str] | None = None,
    ) -> None:
        self.runner = runner
        self.options = options or WorkflowRuntimeOptions()
        self.saved_workflows = dict(saved_workflows or {})

    def run(
        self,
        *,
        script: str,
        task: str,
        artifacts_dir: str | Path,
        args: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        name: str = "workflow",
    ) -> WorkflowRunResult:
        artifacts = Path(artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        script_path = artifacts / "workflow.js"
        journal_path = artifacts / "workflow_journal.jsonl"
        result_path = artifacts / "workflow_result.json"
        script_path.write_text(script, encoding="utf-8")

        journal = WorkflowJournal(journal_path)
        script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()[:16]
        journal.append(
            "workflow_started",
            name=name,
            script_hash=script_hash,
            task=task,
            args=dict(args or {}),
        )

        raw_result, calls = self._run_node(
            script=script,
            task=task,
            artifacts_dir=artifacts,
            journal=journal,
            script_hash=script_hash,
            args=dict(args or {}),
            budget=dict(budget or {}),
            filename=str(script_path),
        )
        output = _result_output(raw_result)
        result = WorkflowRunResult(
            output=output,
            raw_result=raw_result,
            script_path=script_path,
            journal_path=journal_path,
            result_path=result_path,
            artifacts_dir=artifacts,
            agent_calls=tuple(calls),
        )
        result_path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        journal.append("workflow_completed", output=output, result=raw_result)
        return result

    def _run_node(
        self,
        *,
        script: str,
        task: str,
        artifacts_dir: Path,
        journal: WorkflowJournal,
        script_hash: str,
        args: dict[str, Any],
        budget: dict[str, Any],
        filename: str,
    ) -> tuple[Any, list[dict[str, Any]]]:
        command = [
            *self.options.process_prefix,
            *_node_command(
                self.options.node_binary,
                enforce_permissions=self.options.enforce_node_permissions,
            ),
        ]
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=artifacts_dir,
            env=_node_env(),
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError("failed to open node workflow runtime pipes")

        messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        stderr_lines: list[str] = []
        reader = threading.Thread(
            target=_read_stdout, args=(proc.stdout, messages), daemon=True
        )
        err_reader = threading.Thread(
            target=_read_stderr, args=(proc.stderr, stderr_lines), daemon=True
        )
        reader.start()
        err_reader.start()

        start = {
            "script": script,
            "args": {"task": task, **args},
            "budget": budget,
            "maxConcurrency": self.options.max_concurrency,
            "filename": filename,
            "fetchEnabled": False,
        }
        _send(proc, start)

        calls: list[dict[str, Any]] = []
        final_result: Any = None
        final_error = ""
        deadline = (
            time.monotonic() + self.options.timeout_seconds
            if self.options.timeout_seconds is not None
            else None
        )
        agent_count = 0
        futures: dict[Future[dict[str, Any]], str] = {}
        pool: _DaemonThreadPool | None = None
        pool_closed = False

        try:
            pool = _DaemonThreadPool(max_workers=max(1, self.options.max_concurrency))
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    _abort_process(proc)
                    _cancel_futures(futures)
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool_closed = True
                    raise TimeoutError(
                        "dynamic workflow timed out after "
                        f"{self.options.timeout_seconds}s"
                    )

                _flush_completed(proc, futures)

                if final_error:
                    _abort_process(proc)
                    _cancel_futures(futures)
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool_closed = True
                    raise RuntimeError(final_error)
                if final_result is not None:
                    _flush_completed(proc, futures)
                    if not futures:
                        break

                if proc.poll() is not None and messages.empty() and not futures:
                    break

                try:
                    message = messages.get(timeout=0.05)
                except queue.Empty:
                    continue

                kind = message.get("type")
                if kind == "event":
                    event = dict(message.get("event") or {})
                    event_kind = str(event.pop("kind", "workflow_event"))
                    journal.append(event_kind, **event)
                    continue

                if kind == "request":
                    req_id = str(message.get("id") or "")
                    method = str(message.get("method") or "")
                    params = dict(message.get("params") or {})
                    if method == "agent":
                        agent_count += 1
                        if agent_count > self.options.max_agents:
                            _send_error(
                                proc,
                                req_id,
                                "dynamic workflow agent cap exceeded",
                            )
                            continue
                        future = pool.submit(
                            self._handle_agent_request,
                            params=params,
                            artifacts_dir=artifacts_dir,
                            journal=journal,
                            script_hash=script_hash,
                            calls=calls,
                        )
                    elif method == "workflow":
                        future = pool.submit(
                            self._handle_workflow_request,
                            params=params,
                            artifacts_dir=artifacts_dir,
                        )
                    else:
                        _send_error(
                            proc,
                            req_id,
                            f"unsupported workflow method {method!r}",
                        )
                        continue
                    futures[future] = req_id
                    continue

                if kind == "result":
                    final_result = message.get("result")
                    continue

                if kind == "error":
                    final_error = str(message.get("error") or "workflow failed")
                    continue

            pool.shutdown(wait=True)
            pool_closed = True
            exit_code = _finish_process(proc)
            if exit_code != 0 and final_result is None:
                stderr = "".join(stderr_lines).strip()
                raise RuntimeError(
                    f"node workflow runtime exited {exit_code}: {stderr}"
                )
            if final_result is None:
                stderr = "".join(stderr_lines).strip()
                raise RuntimeError(f"workflow did not return a result: {stderr}")
            return final_result, calls
        finally:
            if pool is not None and not pool_closed:
                pool.shutdown(wait=False, cancel_futures=True)
            _close_streams(proc)

    def _handle_agent_request(
        self,
        *,
        params: Mapping[str, Any],
        artifacts_dir: Path,
        journal: WorkflowJournal,
        script_hash: str,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = str(params.get("prompt") or "")
        options = AgentCallOptions.from_mapping(
            params.get("options") if isinstance(params.get("options"), Mapping) else {}
        )
        call_id = str(params.get("call_id") or options.cache_key or "agent")
        phase = str(params.get("phase") or "")
        stable_key = options.cache_key or call_id
        cache_key = f"{script_hash}:{stable_key}"
        cached = journal.cached(cache_key)
        if cached is not None:
            cached["reused"] = True
            journal.append(
                "agent_reused",
                call_id=call_id,
                cache_key=cache_key,
                phase=phase,
                name=cached.get("name", options.name),
            )
            calls.append(dict(cached))
            return cached

        journal.append(
            "agent_started",
            call_id=call_id,
            cache_key=cache_key,
            phase=phase,
            name=options.name,
            prompt=prompt,
            options=_jsonable_options(options),
        )
        try:
            result = self.runner.run_agent(
                prompt,
                options=options,
                call_id=call_id,
                phase=phase,
                artifacts_dir=artifacts_dir,
            ).as_dict()
        except Exception as exc:
            journal.append(
                "agent_failed",
                call_id=call_id,
                cache_key=cache_key,
                phase=phase,
                name=options.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        journal.append(
            "agent_completed",
            call_id=call_id,
            cache_key=cache_key,
            phase=phase,
            name=result.get("name", options.name),
            result=result,
        )
        calls.append(dict(result))
        return result

    def _handle_workflow_request(
        self, *, params: Mapping[str, Any], artifacts_dir: Path
    ) -> dict[str, Any]:
        name = str(params.get("name") or "")
        if name not in self.saved_workflows:
            raise RuntimeError(f"saved workflow {name!r} is not available")
        raw_args = params.get("args")
        nested_args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
        nested = self.run(
            script=self.saved_workflows[name],
            task=str(nested_args.get("task") or name),
            artifacts_dir=artifacts_dir / "workflows" / _safe_part(name),
            args=nested_args,
            name=name,
        )
        return nested.as_dict()


def _flush_completed(
    proc: subprocess.Popen[str], futures: dict[Future[dict[str, Any]], str]
) -> None:
    for future, req_id in list(futures.items()):
        if not future.done():
            continue
        futures.pop(future)
        try:
            result = future.result()
        except Exception as exc:
            _send_error(proc, req_id, f"{type(exc).__name__}: {exc}")
        else:
            _send(proc, {"id": req_id, "ok": True, "result": result})


class _DaemonThreadPool:
    """Small executor whose abandoned workers do not block process shutdown.

    Python's standard ThreadPoolExecutor registers every worker with an
    interpreter-exit hook that joins the thread even after shutdown(wait=False).
    That defeats the workflow deadline when an agent call is stuck. Dynamic
    workflows cannot forcibly stop an in-flight Python call, so timed-out calls
    run only on daemon threads while queued calls are cancelled. Normal runs
    still join every worker before returning.
    """

    def __init__(self, *, max_workers: int) -> None:
        self._tasks: queue.Queue[
            tuple[Future[Any], Callable[..., Any], tuple[Any, ...], dict[str, Any]]
            | None
        ] = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"dynamic-workflow-{index}",
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule work after shutdown")
            self._tasks.put((future, function, args, kwargs))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool = False) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_futures:
                    self._cancel_queued()
                for _thread in self._threads:
                    self._tasks.put(None)
        if wait:
            for thread in self._threads:
                thread.join()

    def _cancel_queued(self) -> None:
        while True:
            try:
                task = self._tasks.get_nowait()
            except queue.Empty:
                return
            try:
                if task is not None:
                    task[0].cancel()
            finally:
                self._tasks.task_done()

    def _worker(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                if task is None:
                    return
                future, function, args, kwargs = task
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._tasks.task_done()


def _cancel_futures(futures: dict[Future[dict[str, Any]], str]) -> None:
    for future in futures:
        future.cancel()
    futures.clear()


def _abort_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _finish_process(proc: subprocess.Popen[str]) -> int:
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    if proc.poll() is None:
        try:
            return proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                return proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait(timeout=1)
    return int(proc.returncode or 0)


def _close_streams(proc: subprocess.Popen[str]) -> None:
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _send(proc: subprocess.Popen[str], payload: Mapping[str, Any]) -> None:
    if proc.stdin is None:
        raise RuntimeError("node workflow runtime stdin is closed")
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _send_error(proc: subprocess.Popen[str], req_id: str, error: str) -> None:
    _send(proc, {"id": req_id, "ok": False, "error": error})


def _node_command(node_binary: str, *, enforce_permissions: bool) -> list[str]:
    node = _resolve_node_binary(node_binary)
    command = [node]
    if enforce_permissions:
        # The vm context is not a sandbox; constrain host capabilities on the child.
        command.append(_node_permission_flag(node))
    command.extend(["-e", _NODE_RUNTIME_SOURCE])
    return command


def _resolve_node_binary(node_binary: str) -> str:
    resolved = shutil.which(node_binary)
    if resolved is None:
        raise RuntimeError(f"node binary not found: {node_binary!r}")
    return resolved


@lru_cache(maxsize=8)
def _node_permission_flag(node_binary: str) -> str:
    for flag in ("--permission", "--experimental-permission"):
        try:
            result = subprocess.run(
                [node_binary, flag, "-e", ""],
                check=False,
                env=_node_env(),
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return flag
    raise RuntimeError(
        "dynamic workflows require Node's permission model; install a Node "
        "version that supports --permission, or only run trusted scripts with "
        "WorkflowRuntimeOptions(enforce_node_permissions=False)"
    )


def _node_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in ("PATH", "SystemRoot", "WINDIR", "PATHEXT"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def _read_stdout(stream: Any, messages: "queue.Queue[dict[str, Any]]") -> None:
    for line in stream:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            messages.put({"type": "error", "error": f"invalid node message: {line}"})
            continue
        if isinstance(parsed, dict):
            messages.put(parsed)


def _read_stderr(stream: Any, lines: list[str]) -> None:
    for line in stream:
        lines.append(line)


def _result_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        for key in ("output", "answer", "final", "text"):
            value = raw.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def _jsonable_options(options: AgentCallOptions) -> dict[str, Any]:
    return {
        "name": options.name,
        "role": options.role,
        "system_prompt": options.system_prompt,
        "model": options.model,
        "tools": list(options.tools),
        "max_turns": options.max_turns,
        "timeout_seconds": options.timeout_seconds,
        "worktree": options.worktree,
        "schema": dict(options.schema or {}),
        "cache_key": options.cache_key,
    }


def _safe_part(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)
    return safe or "workflow"


_NODE_RUNTIME_SOURCE = r"""
const readline = require("readline");
const vm = require("vm");

const pending = new Map();
let nextRequestId = 1;
let nextAgentId = 1;
let currentPhase = "";

function write(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function send(method, params) {
  const id = "req_" + nextRequestId++;
  write({ type: "request", id, method, params });
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
}

function serialize(value) {
  if (value === undefined) return null;
  if (value instanceof Error) {
    return { error: value.message, name: value.name, stack: value.stack || "" };
  }
  return JSON.parse(JSON.stringify(value));
}

function phase(name) {
  currentPhase = String(name || "");
  write({ type: "event", event: { kind: "phase_started", phase: currentPhase } });
}

function log(message) {
  write({ type: "event", event: { kind: "log", phase: currentPhase, message: String(message || "") } });
}

function agent(prompt, opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  const callId = String(options.callId || options.call_id || options.cacheKey || ("agent_" + nextAgentId++));
  return send("agent", {
    prompt: String(prompt || ""),
    options,
    call_id: callId,
    phase: currentPhase
  });
}

async function parallel(thunks, opts = {}) {
  if (!Array.isArray(thunks)) throw new Error("parallel expects an array");
  const max = Math.max(1, Number(opts.maxConcurrency || globalThis.__maxConcurrency || thunks.length || 1));
  const results = new Array(thunks.length);
  let next = 0;
  async function worker() {
    while (next < thunks.length) {
      const index = next++;
      const item = thunks[index];
      results[index] = typeof item === "function" ? await item() : await item;
    }
  }
  const workers = [];
  for (let i = 0; i < Math.min(max, thunks.length); i++) workers.push(worker());
  await Promise.all(workers);
  return results;
}

async function pipeline(items, ...stages) {
  let opts = {};
  if (stages.length && typeof stages[stages.length - 1] === "object" && typeof stages[stages.length - 1] !== "function") {
    opts = stages.pop();
  }
  return parallel(items.map((item) => async () => {
    let value = item;
    for (const stage of stages) value = await stage(value);
    return value;
  }), opts);
}

function workflow(name, workflowArgs = {}) {
  return send("workflow", { name: String(name || ""), args: workflowArgs || {} });
}

async function run(start) {
  const sandbox = {
    JSON, Math, Array, Object, String, Number, Boolean, Date, Promise,
    setTimeout, clearTimeout,
    args: start.args || {},
    budget: start.budget || {},
    agent, parallel, pipeline, workflow, phase, log,
    console: { log: (...parts) => log(parts.join(" ")) },
    __maxConcurrency: start.maxConcurrency || 16
  };
  if (start.fetchEnabled) sandbox.fetch = fetch;
  const context = vm.createContext(sandbox, { name: "simple-agent-lab-workflow" });
  const source = `"use strict";\n(async () => {\n${start.script}\n})()`;
  const compiled = new vm.Script(source, { filename: start.filename || "workflow.js" });
  const result = await compiled.runInContext(context);
  write({ type: "result", result: serialize(result) });
}

let started = false;
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch (err) {
    write({ type: "error", error: "invalid JSON from bridge: " + err.message });
    return;
  }
  if (!started) {
    started = true;
    run(msg).catch((err) => write({ type: "error", error: err && err.stack ? err.stack : String(err) }));
    return;
  }
  const waiter = pending.get(msg.id);
  if (!waiter) return;
  pending.delete(msg.id);
  if (msg.ok) waiter.resolve(msg.result);
  else waiter.reject(new Error(msg.error || "bridge request failed"));
});
"""
