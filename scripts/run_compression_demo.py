"""Watch context compression work over a real agent run.

Two scenarios, both through the real runtime (real ``run()`` loop, real tool
execution, the policy consulted before every model request, real trace
events). Only the "model" is scripted — no provider is called — so the demo is
deterministic and needs no API key.

    uv run python scripts/run_compression_demo.py

Both use one `ContextPolicy(strategy=TieredStrategy((ToolCompact, Summarize)))`:

- Scenario A (tool-heavy): the agent reads big files. The cheap rule-based
  `ToolCompactStrategy` folds older tool exchanges every turn, so the active
  context stays bounded instead of growing unbounded — the "sawtooth". The LLM
  stage stays dormant (first-applicable: the cheap stage suffices).
- Scenario B (text-heavy, no tool exchanges): `ToolCompactStrategy` has nothing
  to fold, so the tier falls through to `SummarizeStrategy`, which runs the
  (scripted) compressor. This exercises the LLM fallback path.
"""

from __future__ import annotations

from simple_agent_lab import (
    Agent,
    ContextCompressionEvent,
    ContextPolicy,
    Message,
    ModelRequestEvent,
    State,
    SummarizeStrategy,
    TieredStrategy,
    ToolCompactStrategy,
    assistant_message,
    message_text,
    run,
)
from simple_agent_lab.messages import TextBlock, ToolCallBlock
from simple_agent_lab.tools import AgentTool, ToolResult, text_result


def _policy() -> ContextPolicy:
    """One tiered strategy in the single `strategy` slot: cheap fold, LLM fallback."""

    def compressor(visible: list[Message]) -> Message:
        del visible
        return assistant_message(
            "Summary of earlier context: key facts and decisions retained.",
            sender="compressor",
            target="runtime",
            kind="final",
        )

    return ContextPolicy(
        strategy=TieredStrategy(
            (
                ToolCompactStrategy(threshold_tokens=900, keep_recent_exchanges=1),
                SummarizeStrategy(
                    compressor=Agent("compressor", compressor),
                    threshold_tokens=900,
                    keep_recent=2,
                ),
            )
        )
    )


def _report(label: str, state: State, events: list) -> None:
    print(f"\n=== {label} ===")
    request = 0
    for event in events:
        if isinstance(event, ModelRequestEvent):
            request += 1
            tokens = event.context_view["estimated_tokens"]
            print(
                f"  request #{request}: ~{tokens:>4} tokens "
                f"({event.visible_count} visible msgs)"
            )
        elif isinstance(event, ContextCompressionEvent):
            replacement = message_text(state.messages[event.summary_message_index])
            print(
                f"   -> compaction: dropped {len(event.compressed_message_indices)} "
                f"msg(s), {event.before_tokens}->{event.after_tokens} tokens, "
                f"replacement={replacement[:42]!r}"
            )
    comps = sum(isinstance(e, ContextCompressionEvent) for e in events)
    print(f"  compactions fired: {comps}")


# --- Scenario A: tool-heavy, stage-1 (ToolCompact) handles it -------------
def _read_file(call_id, args, abort, on_update) -> ToolResult:
    del call_id, abort, on_update
    path = args["path"]
    body = f"# {path}\n" + "\n".join(
        f"line {i:03d}: " + "some realistic source code here " * 4 for i in range(45)
    )
    return text_result(body)


def scenario_tool_heavy() -> None:
    read_tool = AgentTool(
        name="read_file",
        description="Read a source file.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=_read_file,
        execution_mode="sequential",
    )
    files = ["core.py", "state.py", "context_view.py", "compression/runtime.py"]
    turn = {"n": 0}

    def brain(visible: list[Message]) -> Message:
        del visible
        index = turn["n"]
        turn["n"] += 1
        if index < len(files):
            return assistant_message(
                [
                    TextBlock(f"Reading {files[index]}."),
                    ToolCallBlock(f"call_{index}", "read_file", {"path": files[index]}),
                ],
                sender="general-purpose",
                target="user",
                kind="step",
            )
        return assistant_message(
            "Done exploring.", sender="general-purpose", target="user", kind="final"
        )

    state = State("Explore the runtime and summarize how it works.")
    state.send("task", "user", "general-purpose", state.task)
    agent = Agent(
        "general-purpose", brain, tools=(read_tool,), context_policy=_policy()
    )
    events = list(run(agent, state, max_turns=8))
    _report("A: tool-heavy -> ToolCompactStrategy folds (sawtooth)", state, events)


# --- Scenario B: text-heavy, falls through to stage-2 (Summarize) ---------
def scenario_text_heavy() -> None:
    turn = {"n": 0}

    def brain(visible: list[Message]) -> Message:
        del visible
        index = turn["n"]
        turn["n"] += 1
        if index < 4:
            return assistant_message(
                "Here is a long analysis paragraph. " + "elaboration " * 60,
                sender="analyst",
                target="user",
                kind="step",
            )
        return assistant_message("Done.", sender="analyst", target="user", kind="final")

    state = State("Write a long multi-part analysis.")
    state.send("task", "user", "analyst", state.task)
    agent = Agent("analyst", brain, context_policy=_policy())
    events = list(run(agent, state, max_turns=8))
    _report("B: text-heavy -> SummarizeStrategy fallback fires", state, events)


def main() -> None:
    scenario_tool_heavy()
    scenario_text_heavy()


if __name__ == "__main__":
    main()
