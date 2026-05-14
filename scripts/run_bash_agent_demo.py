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


OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_REQUIRED_ENVS = (OPENAI_MODEL_ENV, OPENAI_AUTH_ENV)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (  # noqa: E402
    AssistantMessage,
    Event,
    Message,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    message_text,
    print_trace,
    text_of,
    tool_results_of,
)
from simple_agent_lab.agents.bash import (  # noqa: E402
    BASH_AGENT_SYSTEM_PROMPT,
    make_bash_agent,
)
from simple_agent_lab.llm import Provider  # noqa: E402


DEFAULT_BASH_DEMO_COMMAND = (
    "pwd && find src/simple_agent_lab -maxdepth 1 -type f -name '*.py' | sort"
)


def build_fake_provider() -> Provider:
    return Provider(id="fake", api="fake", model="fake-model")


def bash_task_for_command(command: str) -> str:
    return f"Use bash to run command: `{command}`"


def build_openai_provider() -> Provider:
    env_values = {
        OPENAI_MODEL_ENV: (os.environ.get(OPENAI_MODEL_ENV) or "").strip(),
        OPENAI_BASE_URL_ENV: (os.environ.get(OPENAI_BASE_URL_ENV) or "").strip(),
        OPENAI_AUTH_ENV: (os.environ.get(OPENAI_AUTH_ENV) or "").strip(),
    }
    model = env_values[OPENAI_MODEL_ENV]
    base_url = env_values[OPENAI_BASE_URL_ENV] or None
    auth_token = env_values[OPENAI_AUTH_ENV]

    missing = [name for name in OPENAI_REQUIRED_ENVS if not env_values[name]]
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
    elif isinstance(event, ToolExecutionStartEvent):
        print(f"  [      tool] start {event.tool_name}")
    elif isinstance(event, ToolExecutionEndEvent):
        print(f"  [      tool] end {event.tool_name} error={event.is_error}")


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
            'fine-tuning JSONL line ({"messages": ..., "tools": ...}) at PATH.'
        ),
    )
    args = parser.parse_args()

    task = args.task
    command = args.command
    if task is None and command is None:
        command = DEFAULT_BASH_DEMO_COMMAND
    elif task is not None and command is not None:
        parser.error("--task and --command are mutually exclusive")

    provider = (
        build_openai_provider() if args.provider == "openai" else build_fake_provider()
    )

    print(f"=== bash-use agent (provider={args.provider}) ===")
    resolved_task = task or bash_task_for_command(command or DEFAULT_BASH_DEMO_COMMAND)
    agent = make_bash_agent(provider, cwd=ROOT)
    state, events = agent.run(resolved_task, max_turns=3)
    for event in events:
        print_live_event(event)

    final = next(
        message for message in reversed(state.messages) if message.kind == "final"
    )
    print("\n=== final ===")
    print(full_message_text(final))

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(state, raw=args.raw)

    if args.save_trace:
        from simple_agent_lab import append_openai_training_record  # noqa: E402
        from simple_agent_lab.tools.bash import make_bash_tool  # noqa: E402

        out = append_openai_training_record(
            state,
            args.save_trace,
            tools=[make_bash_tool(cwd=ROOT)],
            system_prompt=BASH_AGENT_SYSTEM_PROMPT,
        )
        print(f"\n=== saved training record to {out} ===")


if __name__ == "__main__":
    main()
