"""Run the tiny bash-use agent demo.

Examples:

    uv run python -m scripts.run_bash_agent_demo
    uv run python -m scripts.run_bash_agent_demo --command "printf 'hello\\n'"

    # Real end-to-end run against an OpenAI-compatible chat endpoint.
    # Reads OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_AUTH_TOKEN from the env.
    uv run --with openai python -m scripts.run_bash_agent_demo \\
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
from pathlib import Path

from simple_long_horizon_agent import (
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
from simple_long_horizon_agent.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    make_bash_agent,
)
from simple_long_horizon_agent.llm import Provider
from simple_long_horizon_agent.llm.env import FAKE_PROVIDER, provider_from_env


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASH_DEMO_COMMAND = (
    "pwd && find src/simple_long_horizon_agent -maxdepth 1 -type f -name '*.py' | sort"
)


def build_fake_provider() -> Provider:
    return FAKE_PROVIDER


def bash_task_for_command(command: str) -> str:
    return f"Use bash to run command: `{command}`"


def build_openai_provider() -> Provider:
    # Single source of truth in `simple_long_horizon_agent.llm.env`; `reexport_auth`
    # strips the token back into os.environ for the adapter to read.
    return provider_from_env(label="--provider openai", reexport_auth=True)


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
        from simple_long_horizon_agent import append_openai_training_record  # noqa: E402
        from simple_long_horizon_agent.tools.bash import make_bash_tool  # noqa: E402

        out = append_openai_training_record(
            state,
            args.save_trace,
            tools=[make_bash_tool(cwd=ROOT)],
            system_prompt=BASH_AGENT_SYSTEM_PROMPT,
        )
        print(f"\n=== saved training record to {out} ===")


if __name__ == "__main__":
    main()
