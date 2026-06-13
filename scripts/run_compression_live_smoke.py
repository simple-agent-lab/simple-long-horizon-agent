"""Live end-to-end smoke: agent-controlled compaction + recall, real model.

Opt-in (never part of CI): requires the same env vars the TUI gateway and
bash demo use — ``OPENAI_MODEL`` plus ``OPENAI_BASE_URL`` / ``OPENAI_AUTH_TOKEN``
as the endpoint needs. A real model drives the full loop end to end:

1. it reads two documents through a tool (each readable only once);
2. it calls ``compact`` with a summary that deliberately omits the access
   code, so the code leaves its active context;
3. the task then demands the exact code — recoverable only through the
   ``recall`` tool reading the cited transcript indices back.

The script checks the three links of the chain (a compression event with the
agent's summary and a source citation, a recall call, the code in the final
answer) and exits non-zero if any breaks.

    uv run python scripts/run_compression_live_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (  # noqa: E402
    ContextCompressionEvent,
    ContextPolicy,
    State,
    ToolExecutionStartEvent,
    make_compact_control,
    make_llm_agent,
    message_text,
    run,
    text_of,
)
from simple_agent_lab.llm import Provider  # noqa: E402
from simple_agent_lab.llm.env import provider_from_env  # noqa: E402
from simple_agent_lab.tools import (  # noqa: E402
    AgentTool,
    make_recall_tool,
    text_result,
)

ACCESS_CODE = "TANGERINE-7741"

_DOCS = {
    "alpha": (
        "# Doc alpha — deployment runbook\n"
        + "\n".join(f"step {i}: routine deployment detail." for i in range(40))
    ),
    "bravo": (
        "# Doc bravo — credentials appendix\n"
        + "\n".join(f"note {i}: assorted operational detail." for i in range(20))
        + f"\nACCESS CODE: {ACCESS_CODE}\n"
        + "\n".join(f"note {i}: assorted operational detail." for i in range(20, 40))
    ),
}

TASK = (
    "IMPORTANT: first check your context. If it already contains a "
    "compaction summary saying 'Steps 1-3 are done', do NOT start over and "
    "do NOT read any doc — go directly to step 4.\n"
    "Follow these steps exactly, one tool call per step:\n"
    "1. Read doc 'alpha' with read_doc.\n"
    "2. Read doc 'bravo' with read_doc.\n"
    "3. Call compact with summary set to exactly: 'Steps 1-3 are done: alpha "
    "(runbook) and bravo (credentials appendix) were read and may NOT be "
    "read again. Next do step 4: use recall on the transcript indices cited "
    "below to get the bravo ACCESS CODE, then answer.' Do not put any code "
    "in the summary.\n"
    "4. Report the exact ACCESS CODE from doc bravo. It is no longer in "
    "your context after compaction; use the recall tool with the transcript "
    "indices cited by the compaction summary.\n"
    "Finish with one line: FINAL ANSWER: <code>"
)


def _read_doc_tool() -> AgentTool:
    read_count: dict[str, int] = {}

    def execute(call_id, args, abort, on_update):
        del call_id, abort, on_update
        name = str(args.get("name", "")).strip().lower()
        if name not in _DOCS:
            return text_result(
                f"Unknown doc {name!r}; available: {sorted(_DOCS)}", is_error=True
            )
        read_count[name] = read_count.get(name, 0) + 1
        if read_count[name] > 1:
            return text_result(
                f"Doc {name!r} is checked out and cannot be read again.",
                is_error=True,
            )
        return text_result(_DOCS[name])

    return AgentTool(
        name="read_doc",
        description="Read a named document. Each doc can be read only once.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "enum": sorted(_DOCS)}},
            "required": ["name"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _live_provider() -> Provider:
    # Single source of truth: `simple_agent_lab.llm.env`. Missing creds exit 0
    # (SKIP) so this opt-in script is safe to invoke unconditionally.
    def _skip(message: str) -> SystemExit:
        print(f"SKIP: {message}")
        return SystemExit(0)

    return provider_from_env(
        label="OPENAI_MODEL / OPENAI_AUTH_TOKEN", missing_exc=_skip
    )


def main() -> None:
    # keep_recent=0 folds the compact exchange itself too. With a recent
    # assistant kept verbatim, its thinking blocks can smuggle the code past
    # the fold (the model notes the code while deciding to compact) — folding
    # everything makes recall the only way back.
    control = make_compact_control(keep_recent=0)
    state = State(TASK)
    state.send("task", "user", "worker", TASK)
    agent = make_llm_agent(
        name="worker",
        provider=_live_provider(),
        role="You are a careful operations agent. Follow the task steps exactly.",
        tools=(_read_doc_tool(), control.tool, make_recall_tool(state)),
        context_policy=ContextPolicy(strategy=control.strategy),
    )

    tool_calls: list[str] = []
    compressions: list[ContextCompressionEvent] = []
    for event in run(agent, state, max_turns=12):
        if isinstance(event, ToolExecutionStartEvent):
            tool_calls.append(event.tool_name)
            print(f"  tool call: {event.tool_name}")
        elif isinstance(event, ContextCompressionEvent):
            compressions.append(event)
            print(
                f"  compaction: dropped {len(event.compressed_message_indices)} "
                f"msg(s), {event.before_tokens}->{event.after_tokens} tokens"
            )

    final = next(
        (message for message in reversed(state.messages) if message.kind == "final"),
        None,
    )
    final_text = message_text(final) if final is not None else "(no final message)"
    print(f"  final: {final_text!r}")

    checks = {
        "compact applied exactly once": len(compressions) == 1,
        "replacement cites transcript indices": bool(compressions)
        and "[Compressed from transcript messages"
        in text_of(state.messages[compressions[0].summary_message_index].content),
        "code folded out of replacement": bool(compressions)
        and ACCESS_CODE
        not in text_of(state.messages[compressions[0].summary_message_index].content),
        "recall tool used": "recall" in tool_calls,
        "final answer has the code": final is not None and ACCESS_CODE in final_text,
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit(f"live smoke failed: {failed}")
    print("PASS: compact -> citation -> recall -> answer, end to end.")


if __name__ == "__main__":
    main()
