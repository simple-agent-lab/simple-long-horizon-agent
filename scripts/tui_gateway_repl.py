"""A minimal client for the TUI gateway — a text-only stand-in for the eventual
pi-tui front end, and the protocol acceptance check in one place.

It spawns ``python -m simple_agent_lab.tui_gateway.entry``, opens one session,
then either loops interactively (read a line from you, ``prompt.submit`` it, and
pretty-print the event stream — assistant text, thinking, tool runs, results —
until the turn finishes; the session is reused across prompts, so this also
exercises multi-turn history) or, with ``--text``, submits a single prompt
non-interactively and exits.

``--raw`` dumps every JSON-RPC frame to stderr instead of the pretty view. That
is the acceptance-check mode (formerly ``tui_gateway_smoke.py``): spawn the
gateway, run the ``gateway.ready`` → ``session.create`` → ``prompt.submit`` →
``turn.complete`` handshake, and print the frames so you can confirm the backend
half works without Node.

This is deliberately the *shape* the real TS UI will take: a GatewayClient
that frames JSON-RPC over the child's stdio, plus an event → view mapping.

Usage::

    uv run python scripts/tui_gateway_repl.py
    uv run python scripts/tui_gateway_repl.py --provider openai
    printf 'pwd\\nls\\n' | uv run python scripts/tui_gateway_repl.py   # non-interactive
    uv run python scripts/tui_gateway_repl.py --raw --text 'Use bash to run command: `pwd`'  # protocol smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ANSI dim/color helpers — kept tiny; the real UI uses pi-tui themes.
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"


class GatewayClient:
    """Spawns the gateway and frames newline-delimited JSON-RPC over stdio."""

    def __init__(self, provider: str, *, raw: bool = False) -> None:
        self.raw = raw
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "simple_agent_lab.tui_gateway.entry"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert self.proc.stdin and self.proc.stdout and self.proc.stderr
        self.provider = provider
        self._next_id = 0
        # Surface the gateway's stderr (its log channel) dimmed, off to the side.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            sys.stderr.write(f"{DIM}  · {line.rstrip()}{RESET}\n")

    def send_request(self, method: str, params: dict[str, Any]) -> str:
        """Write a request without blocking on its response. Returns the id.

        Use this for streaming methods (``prompt.submit``) whose ack races
        with the event stream; consume both via :meth:`pump_until`."""
        self._next_id += 1
        rid = f"r{self._next_id}"
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return rid

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and block until its correlated response arrives,
        rendering any events that interleave in the meantime.

        Only safe for non-streaming methods (``session.*``) where no
        terminal event races with the response."""
        rid = self.send_request(method, params)
        while True:
            frame = self._read()
            if frame.get("id") == rid:
                if "error" in frame:
                    raise RuntimeError(frame["error"])
                return frame["result"]
            self._render_event(frame)

    def pump_until(self, terminal_types: set[str]) -> str:
        """Render events until one of ``terminal_types`` is seen; return it."""
        while True:
            frame = self._read()
            etype = frame.get("params", {}).get("type")
            if frame.get("id") is not None:
                continue  # a stray response (e.g. prompt.submit's streaming ack)
            self._render_event(frame)
            if etype in terminal_types:
                return etype

    def _write(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin
        line = json.dumps(obj)
        if self.raw:
            sys.stderr.write(f"--> {line}\n")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("gateway closed")
        frame = json.loads(line)
        if self.raw:
            sys.stderr.write(f"<-- {json.dumps(frame, ensure_ascii=False)}\n")
        return frame

    # How many output lines to show before collapsing (pi caps bash at ~20).
    TOOL_PREVIEW_LINES = 16

    def _render_tool_complete(self, payload: dict[str, Any]) -> None:
        is_error = payload.get("is_error")
        color = RED if is_error else CYAN
        lines = payload.get("text", "").splitlines()
        shown = lines[: self.TOOL_PREVIEW_LINES]
        body = "\n".join(f"    {line}" for line in shown)
        if body:
            print(f"{color}{body}{RESET}")
        hidden = len(lines) - len(shown)
        if hidden > 0:
            print(
                f"{DIM}    … (+{hidden} more line{'s' if hidden != 1 else ''}){RESET}"
            )
        # Status footer: ✓/✗ with duration, like pi's "Took 0.4s".
        mark = f"{RED}✗ failed{RESET}" if is_error else f"{GREEN}✓{RESET}"
        dur = payload.get("duration_s")
        took = f" {DIM}· {dur:g}s{RESET}" if isinstance(dur, (int, float)) else ""
        count = payload.get("line_count")
        n = f" {DIM}· {count} line{'s' if count != 1 else ''}{RESET}" if count else ""
        print(f"  {mark}{took}{n}")

    def _render_event(self, frame: dict[str, Any]) -> None:
        if self.raw:
            return  # raw mode already dumped the frame in _read; skip the view
        if frame.get("method") != "event":
            return  # a response frame straggling on the stream — not for the view
        params = frame.get("params", {})
        etype = params.get("type")
        payload = params.get("payload", {})
        if etype == "message.complete":
            if payload.get("is_final"):
                print(f"{GREEN}assistant ›{RESET} {payload['text']}")
            elif payload.get("text"):
                print(f"{DIM}assistant · {payload['text']}{RESET}")
        elif etype == "thinking":
            print(f"{DIM}  (thinking) {payload['text']}{RESET}")
        elif etype == "tool.start":
            # Show the command/args the model chose, up front (pi-style),
            # plus its own short description and a running marker.
            title = payload.get("title") or payload.get("name")
            line = f"{CYAN}  ⏺ {title}{RESET}"
            desc = payload.get("description")
            if desc:
                line += f"  {DIM}{desc}{RESET}"
            print(f"{line} {DIM}…{RESET}")
        elif etype == "tool.complete":
            self._render_tool_complete(payload)
        elif etype == "status.update":
            print(f"{YELLOW}  [status] {payload}{RESET}")
        elif etype == "error":
            print(f"{RED}  [error] {payload.get('message')}{RESET}")
        elif etype in {
            "session.info",
            "gateway.ready",
            "message.start",
            "turn.complete",
        }:
            pass  # lifecycle — no view
        else:
            print(f"{DIM}  [{etype}] {payload}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TUI gateway client")
    parser.add_argument("--provider", default="fake", choices=["fake", "openai"])
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Dump raw JSON-RPC frames to stderr instead of the pretty view "
        "(protocol acceptance check).",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Submit one prompt non-interactively and exit (skips the REPL loop).",
    )
    args = parser.parse_args()

    client = GatewayClient(args.provider, raw=args.raw)
    # Handshake.
    ready = client._read()
    assert ready.get("params", {}).get("type") == "gateway.ready"
    try:
        info = client.request(
            "session.create", {"provider": args.provider, "cwd": str(ROOT)}
        )
    except RuntimeError as exc:
        err = exc.args[0] if exc.args else {}
        msg = err.get("message", err) if isinstance(err, dict) else err
        print(f"{RED}session.create failed: {msg}{RESET}")
        if args.provider == "openai":
            print(
                f"{DIM}--provider openai needs env vars: "
                f"OPENAI_MODEL (required), OPENAI_AUTH_TOKEN (required), "
                f"OPENAI_BASE_URL (optional). e.g.\n"
                f"  OPENAI_MODEL=gpt-4o OPENAI_AUTH_TOKEN=sk-... "
                f"uv run python scripts/tui_gateway_repl.py --provider openai{RESET}"
            )
        client.proc.terminate()
        return 1
    session_id = info["session_id"]
    if not args.raw:
        print(
            f"{DIM}session {session_id} · model={info['info']['model']} · tools={info['info']['tools']}{RESET}"
        )
        print(f"{DIM}Type a prompt (blank line or Ctrl-D to quit).{RESET}\n")

    def submit(text: str) -> None:
        # Fire-and-pump: the {status:"streaming"} ack and the turn's events
        # arrive unordered on the same stream, so we must NOT block waiting
        # for the ack separately — pump_until skips it (it has an id) and
        # stops on the real terminal event. (The real TS client needs this
        # same discipline.)
        client.send_request(
            "prompt.submit",
            {"session_id": session_id, "text": text, "max_turns": args.max_turns},
        )
        client.pump_until({"turn.complete", "error"})

    turn = 0
    if args.text is not None:
        # One-shot: submit a single prompt and exit (the acceptance-check path).
        submit(args.text)
        turn = 1
    else:
        while True:
            try:
                text = input(f"{GREEN}you ›{RESET} ")
            except EOFError:
                break
            if not text.strip():
                break
            turn += 1
            try:
                submit(text)
            except KeyboardInterrupt:
                client.request("session.interrupt", {"session_id": session_id})
                print(f"{YELLOW}  [interrupted]{RESET}")
            print()

    client.request("session.close", {"session_id": session_id})
    client.proc.stdin.close()  # type: ignore[union-attr]
    client.proc.wait(timeout=5)
    if args.raw:
        sys.stderr.write("\n=== gateway protocol check OK ===\n")
    else:
        print(f"{DIM}bye — ran {turn} turn(s) in session {session_id}{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
