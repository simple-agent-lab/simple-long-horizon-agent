"""Session management, method dispatch, and Event → wire translation.

The :class:`Gateway` owns named sessions and routes JSON-RPC requests to
handler methods. A *session* pairs one :class:`~simple_agent_lab.core.Agent`
with a persistent :class:`~simple_agent_lab.state.State` so a conversation
accumulates history across prompts.

Concurrency model: ``prompt.submit`` runs the agent turn on a **worker
thread** and returns ``{"status": "streaming"}`` immediately, so the main
stdin loop stays responsive — that is what lets ``session.interrupt`` flip
the session's abort flag mid-turn. All writes to the transport go through
:meth:`Gateway._send` under a lock, since the worker thread and the main
thread both emit.

Event translation (:meth:`Gateway._pump`) is the heart of the adapter: it
consumes the :data:`~simple_agent_lab.protocols.Event` generator from
:func:`simple_agent_lab.core.run` and maps each runtime event onto a wire
event the UI renders. Phase A emits whole assistant messages
(``message.complete``); there is no token-level ``message.delta`` yet.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from ..agents.bash import make_bash_agent
from ..core import Agent, run
from ..llm import Provider
from ..messages import (
    AssistantMessage,
    TokenUsage,
    is_tool_result_message,
    text_of,
    tool_results_of,
)
from ..protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    MessageEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from ..state import State
from ..tools.bash import BASH_TOOL_NAME
from .transport import Transport, make_error, make_event, make_result

# JSON-RPC error codes. -32xxx are the spec-reserved range; 4xxx are our
# application-level codes (mirrors the Hermes convention).
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_UNKNOWN_SESSION = 4004
ERR_SESSION_BUSY = 4009

DEFAULT_MAX_TURNS = 12

# Env vars for a real OpenAI-compatible chat endpoint (same contract as
# scripts/run_bash_agent_demo.py). Absent → the gateway falls back to the
# deterministic fake provider so it runs with zero secrets.
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"


def _usage_dict(usage: TokenUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def _short(value: Any, limit: int = 60) -> str:
    """One-line, length-capped rendering of an argument value for a header."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_call_view(name: str, args: Any) -> dict[str, str]:
    """A human header for a tool call, mirroring pi's "command up front" style.

    Bash-style tools render as ``$ <command>`` (with a ``(timeout Ns)`` suffix
    when set) and expose the model's own short ``description``; any other tool
    falls back to ``name(k=v, …)``. The UI can use ``title`` directly or render
    its own from the raw ``args`` on the event.
    """
    if not isinstance(args, dict):
        return {"title": name, "description": ""}
    command = args.get("command")
    if name == BASH_TOOL_NAME or command is not None:
        command = str(command or "").strip()
        title = f"$ {command}" if command else name
        timeout = args.get("timeout_seconds")
        if isinstance(timeout, (int, float)) and timeout:
            title += f" (timeout {timeout:g}s)"
        return {"title": title, "description": str(args.get("description", ""))}
    parts = ", ".join(f"{key}={_short(val, 32)}" for key, val in args.items())
    return {"title": f"{name}({parts})", "description": ""}


def _build_provider(kind: str) -> Provider:
    """Resolve a provider from a coarse ``kind`` selector.

    ``"openai"`` reads the same env vars the bash demo uses; anything else
    (including the default) yields the deterministic fake adapter so the
    gateway is runnable end-to-end without credentials.
    """
    if kind == "openai":
        model = (os.environ.get(OPENAI_MODEL_ENV) or "").strip()
        if not model:
            raise ValueError(f"provider 'openai' requires {OPENAI_MODEL_ENV}")
        return Provider(
            id="openai-chat",
            api="openai-chat",
            model=model,
            base_url=(os.environ.get(OPENAI_BASE_URL_ENV) or "").strip() or None,
            api_key_env=OPENAI_AUTH_ENV,
        )
    return Provider(id="fake", api="fake", model="fake-model")


class Session:
    """One agent + its running conversation state.

    ``state`` is lazily created on the first prompt (the runtime's
    :class:`State` is constructed around the seed task). ``abort`` is an
    :class:`threading.Event` the agent loop polls via its ``abort`` callback;
    ``session.interrupt`` sets it. ``busy`` guards against overlapping turns.
    """

    def __init__(
        self, session_id: str, agent: Agent, cwd: str, provider_kind: str
    ) -> None:
        self.id = session_id
        self.agent = agent
        self.cwd = cwd
        self.provider_kind = provider_kind
        self.state: State | None = None
        self.abort = threading.Event()
        self.busy = False
        self.worker: threading.Thread | None = None


class Gateway:
    """Routes JSON-RPC requests to handlers and emits event notifications."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._sessions: dict[str, Session] = {}
        self._session_counter = 0
        # Serializes all frame writes: the prompt worker thread and the main
        # stdin thread both reach the transport.
        self._write_lock = threading.Lock()
        self._methods = {
            "session.create": self._on_session_create,
            "prompt.submit": self._on_prompt_submit,
            "session.interrupt": self._on_session_interrupt,
            "session.close": self._on_session_close,
        }

    # -- transport helpers -------------------------------------------------

    def _send(self, frame: dict[str, Any]) -> bool:
        with self._write_lock:
            return self._transport.write(frame)

    def _emit(
        self, event_type: str, session_id: str | None, payload: dict[str, Any]
    ) -> None:
        self._send(make_event(event_type, session_id, payload))

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, request: Any) -> dict[str, Any] | None:
        """Handle one parsed JSON-RPC request.

        Returns the response frame to write, or ``None`` when the handler
        produces its response/stream asynchronously (``prompt.submit`` runs
        on a worker thread and emits its own frames).
        """
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return make_error(None, ERR_INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            return make_error(request_id, ERR_INVALID_REQUEST, "missing method")
        if not isinstance(params, dict):
            return make_error(
                request_id, ERR_INVALID_PARAMS, "params must be an object"
            )
        handler = self._methods.get(method)
        if handler is None:
            return make_error(
                request_id, ERR_METHOD_NOT_FOUND, f"unknown method: {method}"
            )
        try:
            return handler(request_id, params)
        except _RpcError as exc:
            return make_error(request_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 — surface any handler crash as an RPC error
            return make_error(
                request_id, ERR_INVALID_REQUEST, f"{type(exc).__name__}: {exc}"
            )

    def _require_session(self, params: dict[str, Any]) -> Session:
        session_id = params.get("session_id")
        session = (
            self._sessions.get(session_id) if isinstance(session_id, str) else None
        )
        if session is None:
            raise _RpcError(ERR_UNKNOWN_SESSION, f"unknown session: {session_id!r}")
        return session

    # -- methods -----------------------------------------------------------

    def _on_session_create(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        self._session_counter += 1
        session_id = f"s{self._session_counter}"
        cwd = str(params.get("cwd") or Path.cwd())
        provider_kind = str(params.get("provider") or "fake")
        provider = _build_provider(provider_kind)
        agent = make_bash_agent(provider, cwd=cwd)
        session = Session(session_id, agent, cwd, provider_kind)
        self._sessions[session_id] = session
        info = {
            "model": provider.model,
            "provider": provider_kind,
            "cwd": cwd,
            "tools": [tool.name for tool in agent.tools],
            "agent": agent.name,
        }
        self._emit("session.info", session_id, info)
        return make_result(request_id, {"session_id": session_id, "info": info})

    def _on_prompt_submit(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(params)
        text = params.get("text")
        if not isinstance(text, str) or not text:
            raise _RpcError(
                ERR_INVALID_PARAMS, "prompt.submit requires non-empty 'text'"
            )
        if session.busy:
            raise _RpcError(ERR_SESSION_BUSY, "session is already running a turn")
        max_turns = int(params.get("max_turns") or DEFAULT_MAX_TURNS)

        session.busy = True
        session.abort.clear()
        worker = threading.Thread(
            target=self._run_turn,
            args=(session, text, max_turns),
            name=f"turn-{session.id}",
            daemon=True,
        )
        session.worker = worker
        worker.start()
        return make_result(request_id, {"status": "streaming"})

    def _on_session_interrupt(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(params)
        session.abort.set()
        return make_result(request_id, {"status": "interrupted"})

    def _on_session_close(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._require_session(params)
        session.abort.set()
        self._sessions.pop(session.id, None)
        return make_result(request_id, {"closed": True})

    # -- turn execution ----------------------------------------------------

    def _run_turn(self, session: Session, text: str, max_turns: int) -> None:
        """Worker-thread body: seed the prompt, run the loop, pump events."""
        try:
            agent = session.agent
            if session.state is None:
                # First prompt seeds the State around the task (kind="task"),
                # mirroring Agent.run's seeding.
                session.state = State(task=text)
                session.state.send("task", "user", agent.name, text)
            else:
                # Follow-up user turn appended to the existing conversation.
                session.state.send("message", "user", agent.name, text)
            events = run(
                agent,
                session.state,
                max_turns=max_turns,
                abort=session.abort.is_set,
            )
            self._pump(session, events)
        except Exception as exc:  # noqa: BLE001 — never let a turn crash the gateway
            self._emit("error", session.id, {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            session.busy = False

    def _pump(self, session: Session, events: Any) -> None:
        """Translate the runtime Event stream into wire events.

        ``tools`` accumulates per-call detail across the events of one turn so
        the right data can ride on each wire event: the assistant message that
        *requests* a call (carrying its arguments) arrives before the
        ``tool_execution_start``; the result text arrives last in the
        ``tool_result`` bundle; and the start/end elapsed stamps bracket the
        duration. We stash all of it keyed by ``tool_call_id`` and assemble
        ``tool.start`` / ``tool.complete`` with full context — mirroring how
        the pi coding agent shows the command up front and the duration after.
        """
        sid = session.id
        tools: dict[str, dict[str, Any]] = {}
        for event in events:
            if isinstance(event, MessageEvent):
                self._emit_message(sid, event, tools)
            elif isinstance(event, ToolExecutionStartEvent):
                record = tools.setdefault(event.tool_call_id, {})
                record["name"] = event.tool_name
                record["start"] = event.elapsed
                args = record.get("args", {})
                view = _tool_call_view(event.tool_name, args)
                self._emit(
                    "tool.start",
                    sid,
                    {
                        "tool_call_id": event.tool_call_id,
                        "name": event.tool_name,
                        "args": args,
                        "title": view["title"],
                        "description": view["description"],
                    },
                )
            elif isinstance(event, ToolExecutionUpdateEvent):
                self._emit(
                    "tool.progress",
                    sid,
                    {
                        "tool_call_id": event.tool_call_id,
                        "name": event.tool_name,
                        "text": text_of(event.partial.content),
                    },
                )
            elif isinstance(event, ToolExecutionEndEvent):
                record = tools.setdefault(event.tool_call_id, {})
                start = record.get("start")
                if start is not None:
                    record["duration_s"] = round(event.elapsed - start, 3)
                record["is_error"] = event.is_error
            elif isinstance(event, ContextCompressionEvent):
                self._emit(
                    "status.update",
                    sid,
                    {
                        # `rewrite` is an in-progress field on this event; fall
                        # back to plain compression until it lands.
                        "kind": "rewrite"
                        if getattr(event, "rewrite", False)
                        else "compression",
                        "before_tokens": event.before_tokens,
                        "after_tokens": event.after_tokens,
                    },
                )
            elif isinstance(event, AgentStartEvent):
                self._emit("message.start", sid, {})
            elif isinstance(event, AgentEndEvent):
                self._emit("turn.complete", sid, {"reason": event.reason})
            # Turn boundaries and model request/response events carry no UI
            # surface in phase A; they stay internal to the runtime trace.

    def _emit_message(
        self, sid: str, event: MessageEvent, tools: dict[str, dict[str, Any]]
    ) -> None:
        message = event.message
        if isinstance(message, AssistantMessage):
            for block in message.thinking:
                self._emit(
                    "thinking",
                    sid,
                    {"text": block.text, "redacted": block.redacted},
                )
            # Stash each requested call's arguments so the upcoming
            # tool.start can render the command/path the model chose.
            for call in message.tool_calls:
                tools.setdefault(call.id, {})["args"] = dict(call.arguments)
            self._emit(
                "message.complete",
                sid,
                {
                    "sender": message.sender,
                    "text": text_of(message.content),
                    "kind": message.kind,
                    "is_final": message.kind == "final",
                    "usage": _usage_dict(message.usage),
                    "model": message.model,
                },
            )
        elif is_tool_result_message(message):
            # The tool result text the model will see; one event per block so a
            # parallel-tool bundle renders as separate results. Duration and
            # the model-chosen title come from what we stashed during the turn.
            for block in tool_results_of(message.content):
                record = tools.get(block.tool_call_id, {})
                text = text_of(block.content)
                view = _tool_call_view(block.tool_name, record.get("args", {}))
                self._emit(
                    "tool.complete",
                    sid,
                    {
                        "tool_call_id": block.tool_call_id,
                        "name": block.tool_name,
                        "title": view["title"],
                        "text": text,
                        "line_count": len(text.splitlines()),
                        "is_error": block.is_error,
                        "duration_s": record.get("duration_s"),
                    },
                )
        # User/system messages need no echo: the UI already shows the prompt.


class _RpcError(Exception):
    """Internal: a handler-level failure carrying a JSON-RPC error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
