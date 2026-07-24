"""Run the interactive skills-aware agent on a free-form prompt.

This is the *interactive* edge of agent skills (`run_with_skills`), the
counterpart to the benchmark path. Skills are driven entirely by the prompt:

    /no-skills <task>          -> skills disabled for this run
    <task>                     -> model auto-decides from the menu
    /pdf <task>                -> explicitly invoke the `pdf` skill (body preloaded)
    /systematic-debugging ...  -> explicitly invoke that skill

There are no CLI flags that turn skills on/off or pick a skill — that is the
whole point: the directive lives in the prompt, exactly like a human typing to
the agent. The agent carries a `bash` tool (to run skill scripts / shell) and a
`read` tool (to load `SKILL.md` and references).

Examples:

    uv run python scripts/run_skill_agent_demo.py --provider openai \\
        --cwd .tmp/pdf_test \\
        --save-trace .tmp/pdf_test/trace.jsonl \\
        --task "/pdf Combine page 2 of A.pdf and page 3 of B.pdf into combined.pdf"

Reads OPENAI_MODEL / OPENAI_AUTH_TOKEN / OPENAI_BASE_URL from --dotenv (.env).
By default the menu is scoped to the bundled skill library so the run is a
focused, reproducible comparison; pass --all-skills for full real-user
discovery (also surfaces ~/.agents/skills).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_agent_lab import (
    AssistantMessage,
    Event,
    message_text,
    text_of,
    tool_results_of,
)
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import (
    FAKE_PROVIDER,
    load_dotenv,
    provider_from_env,
)
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.skills import (
    BUNDLED_LIBRARY_DIR,
    SkillRoot,
    default_skill_roots,
    run_with_skills,
)
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool
from simple_agent_lab.trace import run_trace_from_state, write_event_stream

ROOT = Path(__file__).resolve().parents[1]
AGENT_NAME = "skill_agent"
AGENT_ROLE = (
    "You are a capable software agent with a bash tool and a read tool. Use the "
    "available skills when they fit the task: read a skill's SKILL.md, then run "
    "its scripts via bash. Work from evidence and verify your result."
)


def build_openai_provider() -> Provider:
    # Single source of truth: `simple_agent_lab.llm.env`. `reexport_auth` strips
    # the token back into os.environ for the adapter to read.
    return provider_from_env(label="--provider openai", reexport_auth=True)


def print_live_event(event: Event) -> None:
    if event.kind != "message" or event.message is None:
        return
    message = event.message
    if isinstance(message, AssistantMessage) and message.sender == AGENT_NAME:
        print(f"  [{message.sender:>14}] {message_text(message)[:300]}")
    elif message.kind == "tool_result":
        for block in tool_results_of(message.content):
            inner = text_of(block.content).replace("\n", " ")
            if len(inner) > 240:
                inner = inner[:240] + "..."
            tag = "tool*" if block.is_error else block.tool_name
            print(f"  [{tag:>14}] {inner}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, help="Prompt (may start with a /directive)."
    )
    parser.add_argument(
        "--cwd", default=str(ROOT), help="Working dir for bash/read tools."
    )
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--save-trace", default=None, metavar="PATH")
    parser.add_argument(
        "--all-skills",
        action="store_true",
        help="Use full real-user discovery instead of just the bundled library.",
    )
    parser.add_argument(
        "--trace-id",
        default="skill-demo",
        help="trace_id stamped into the saved trace.",
    )
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    if args.provider == "openai":
        load_dotenv(args.dotenv)
        provider = build_openai_provider()
    else:
        provider = FAKE_PROVIDER

    # Scope the menu: bundled library only (focused) or full real-user discovery.
    if args.all_skills:
        roots = default_skill_roots(str(cwd))
    else:
        roots = [SkillRoot(BUNDLED_LIBRARY_DIR, "bundled")]

    agent = make_llm_agent(
        name=AGENT_NAME,
        provider=provider,
        role=AGENT_ROLE,
        tools=[make_bash_tool(cwd=cwd), make_read_tool(cwd=cwd)],
        system_prompt=AGENT_ROLE,
        target="user",
    )

    print(f"=== skill agent (provider={args.provider}, cwd={cwd}) ===")
    print(f"=== task: {args.task[:160]} ===")
    state, events = run_with_skills(
        agent, args.task, roots=roots, cwd=str(cwd), max_turns=args.max_turns
    )
    for event in events:
        print_live_event(event)

    final = next((m for m in reversed(state.messages) if m.kind == "final"), None)
    print("\n=== final ===")
    print(text_of(final.content) if final is not None else "(no final message)")

    if args.save_trace:
        trace = run_trace_from_state(
            state=state,
            trace_id=args.trace_id,
            producer="scripts:run_skill_agent_demo",
            meta={"task": args.task, "cwd": str(cwd)},
        )
        out = Path(args.save_trace)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_event_stream(out, trace)
        print(f"\n=== saved trace to {out} ===")


if __name__ == "__main__":
    main()
