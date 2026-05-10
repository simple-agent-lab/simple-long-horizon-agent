"""Run the tiny bash-use agent demo.

Examples:

    uv run python scripts/run_bash_agent_demo.py
    uv run python scripts/run_bash_agent_demo.py --command "printf 'hello\\n'"

The demo is deterministic: it uses the fake LLM adapter, but still goes through
the real runtime path of model request, tool call, bash execution, tool result,
and final answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (  # noqa: E402
    DEFAULT_BASH_DEMO_COMMAND,
    Event,
    Message,
    last_message,
    message_text,
    print_trace,
    run_bash_agent_demo,
)


def print_live_event(event: Event) -> None:
    if event.kind == "message":
        message = event.message
        if message is not None and message.sender in {"bash_agent", "bash"}:
            print(f"  [{message.sender:>10}] {message_text(message)}")
    elif event.kind == "tool_execution_start":
        print(f"  [      tool] start {event.data.get('tool_name')}")
    elif event.kind == "tool_execution_end":
        print(
            f"  [      tool] end {event.data.get('tool_name')} "
            f"error={event.data.get('is_error')}"
        )


def full_message_text(message: Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        text
        for block in content
        if isinstance((text := getattr(block, "text", "")), str) and text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny bash-use agent demo")
    parser.add_argument(
        "--command",
        default=DEFAULT_BASH_DEMO_COMMAND,
        help="Bash command for the demo agent to run",
    )
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    print("=== bash-use agent ===")
    runtime = run_bash_agent_demo(
        command=args.command,
        cwd=ROOT,
        on_event=print_live_event,
    )

    final = last_message(runtime.state, kind="final")
    print("\n=== final ===")
    print(full_message_text(final))

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(runtime.state)


if __name__ == "__main__":
    main()
