"""Run the tiny bash-use agent demo.

Examples:

    uv run python scripts/run_bash_agent_demo.py
    uv run python scripts/run_bash_agent_demo.py --command "printf 'hello\\n'"

    # Real end-to-end run against an OpenAI-compatible chat endpoint.
    # Reads OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_AUTH_TOKEN from the env.
    uv run --with openai python scripts/run_bash_agent_demo.py \\
        --provider openai \\
        --task "Create a file at /tmp/hello.txt with the content 'hello, world!'."

By default the demo is deterministic: it uses the fake LLM adapter, but still
goes through the real runtime path of model request, tool call, bash execution,
tool result, and final answer. With ``--provider openai`` it instead calls the
configured chat model end-to-end. Use ``--task`` for a free-form prompt
(letting the model pick the command); use ``--command`` to pin the exact bash
command the agent should run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (  # noqa: E402
    AssistantMessage,
    Event,
    Message,
    last_message,
    message_text,
    print_trace,
    text_of,
    tool_results_of,
)
from simple_agent_lab.agents.bash import (  # noqa: E402
    DEFAULT_BASH_DEMO_COMMAND,
    run_bash_agent_demo,
)
from simple_agent_lab.llm import Provider  # noqa: E402


OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"


def build_openai_provider() -> Provider:
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    auth_token = (os.environ.get(OPENAI_AUTH_ENV) or "").strip()

    missing = [
        name
        for name, value in (
            ("OPENAI_MODEL", model),
            (OPENAI_AUTH_ENV, auth_token),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required env vars for --provider openai: " + ", ".join(missing)
        )

    # Re-export the stripped token so the adapter (which reads os.environ
    # directly) gets a clean value even if the user's export had stray spaces.
    os.environ[OPENAI_AUTH_ENV] = auth_token

    return Provider(
        id="openai-chat",
        api="openai-chat",
        model=model,
        base_url=base_url,
        api_key_env=OPENAI_AUTH_ENV,
    )


def print_live_event(event: Event) -> None:
    if event.kind == "message":
        message = event.message
        if message is None:
            return
        if isinstance(message, AssistantMessage) and message.sender == "bash_agent":
            for thinking_block in message.thinking:
                preview = thinking_block.text.replace("\n", " ")
                if len(preview) > 240:
                    preview = preview[:240] + "..."
                tag = "thinking*" if thinking_block.redacted else "thinking"
                print(f"  [{tag:>10}] {preview}")
            print(f"  [{message.sender:>10}] {message_text(message)}")
        elif message.kind == "tool_result":
            for block in tool_results_of(message.content):
                inner = text_of(block.content).replace("\n", " ")
                if len(inner) > 240:
                    inner = inner[:240] + "..."
                tag = "tool*" if block.is_error else block.tool_name
                print(f"  [{tag:>10}] {inner}")
    elif event.kind == "tool_execution_start":
        print(f"  [      tool] start {event.data.get('tool_name')}")
    elif event.kind == "tool_execution_end":
        print(
            f"  [      tool] end {event.data.get('tool_name')} "
            f"error={event.data.get('is_error')}"
        )


def full_message_text(message: Message) -> str:
    return text_of(message.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny bash-use agent demo")
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "High-level task text for the agent; the model chooses the bash "
            "command. Mutually exclusive with --command (--task wins)."
        ),
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Exact bash command for the demo agent to run.",
    )
    parser.add_argument(
        "--provider",
        choices=["fake", "openai"],
        default="fake",
        help="LLM provider: 'fake' (deterministic, default) or 'openai' (real chat call)",
    )
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also dump each turn's `raw` payload — request snapshot + SDK response dump.",
    )
    parser.add_argument(
        "--save-trace",
        default=None,
        metavar="PATH",
        help=(
            "After the run, append the conversation as one OpenAI Chat "
            "fine-tuning JSONL line ({\"messages\": ..., \"tools\": ...}) at PATH."
        ),
    )
    args = parser.parse_args()

    task = args.task
    command = args.command
    if task is None and command is None:
        command = DEFAULT_BASH_DEMO_COMMAND
    elif task is not None and command is not None:
        parser.error("--task and --command are mutually exclusive")

    provider = build_openai_provider() if args.provider == "openai" else None

    print(f"=== bash-use agent (provider={args.provider}) ===")
    runtime = run_bash_agent_demo(
        task=task,
        command=command,
        cwd=ROOT,
        provider=provider,
        on_event=print_live_event,
    )

    final = last_message(runtime.state, kind="final")
    print("\n=== final ===")
    print(full_message_text(final))

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(runtime.state, raw=args.raw)

    if args.save_trace:
        from simple_agent_lab import append_openai_training_record  # noqa: E402
        from simple_agent_lab.agents.bash import (  # noqa: E402
            BASH_AGENT_SYSTEM_PROMPT,
            make_bash_tool,
        )

        out = append_openai_training_record(
            runtime.state,
            args.save_trace,
            tools=[make_bash_tool(cwd=ROOT)],
            system_prompt=BASH_AGENT_SYSTEM_PROMPT,
        )
        print(f"\n=== saved training record to {out} ===")


if __name__ == "__main__":
    main()
