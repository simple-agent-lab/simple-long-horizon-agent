"""Drive the TUI gateway over stdio with no TS/Node — a pure-Python stand-in
for the eventual UI.

This is the phase-A acceptance check: it spawns
``python -m simple_agent_lab.tui_gateway.entry`` exactly as a real UI would,
performs the ``gateway.ready`` → ``session.create`` → ``prompt.submit``
handshake, and prints every JSON-RPC frame the gateway emits until the turn
completes. If you see ``gateway.ready``, a ``session.info``, the
``tool.start`` / ``tool.complete`` pair, a ``message.complete`` with
``is_final: true``, and a final ``turn.complete``, the backend half works.

Usage::

    uv run python scripts/tui_gateway_smoke.py
    uv run python scripts/tui_gateway_smoke.py --provider openai \\
        --text "List the python files in src/simple_agent_lab"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _reader(stderr) -> None:
    """Echo the gateway's stderr (its log channel) with a prefix."""
    for line in stderr:
        sys.stderr.write(f"  [gateway.stderr] {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Text smoke test for the TUI gateway")
    parser.add_argument("--provider", default="fake", choices=["fake", "openai"])
    parser.add_argument(
        "--text",
        default="Use bash to run command: `pwd`",
        help="The user prompt to submit.",
    )
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args()

    proc = subprocess.Popen(
        [sys.executable, "-m", "simple_agent_lab.tui_gateway.entry"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.stdin and proc.stdout and proc.stderr
    threading.Thread(target=_reader, args=(proc.stderr,), daemon=True).start()

    def send(obj: dict) -> None:
        line = json.dumps(obj)
        sys.stderr.write(f"\n--> {line}\n")
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def read_frame() -> dict:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("gateway closed stdout unexpectedly")
        frame = json.loads(line)
        sys.stderr.write(f"<-- {json.dumps(frame, ensure_ascii=False)}\n")
        return frame

    # 1. Handshake: expect gateway.ready.
    ready = read_frame()
    assert ready.get("params", {}).get("type") == "gateway.ready", ready

    # 2. session.create.
    send(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "session.create",
            "params": {"provider": args.provider, "cwd": str(ROOT)},
        }
    )
    session_id = None
    while session_id is None:
        frame = read_frame()
        if frame.get("id") == "r1":
            session_id = frame["result"]["session_id"]

    # 3. prompt.submit, then pump events until turn.complete.
    send(
        {
            "jsonrpc": "2.0",
            "id": "r2",
            "method": "prompt.submit",
            "params": {
                "session_id": session_id,
                "text": args.text,
                "max_turns": args.max_turns,
            },
        }
    )
    while True:
        frame = read_frame()
        if frame.get("params", {}).get("type") == "turn.complete":
            break
        if frame.get("params", {}).get("type") == "error":
            break

    # 4. Clean shutdown.
    send(
        {
            "jsonrpc": "2.0",
            "id": "r3",
            "method": "session.close",
            "params": {"session_id": session_id},
        }
    )
    read_frame()
    proc.stdin.close()
    proc.wait(timeout=5)
    sys.stderr.write("\n=== smoke test OK ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
