"""Realistic transcripts for compression evals.

A `Scenario` is a named transcript builder plus the metadata the metrics
layer needs: the durable facts planted in the *compressible* region (so a
summary that drops them is measurably wrong) and the recent-tail size the
matching strategy should keep verbatim.

Builders are model-free and deterministic — call `scenario.build()` to get a
fresh `State`. Facts are planted in the older messages (before the recent
tail), each carrying a unique `needle` token so fact retention can be scored
by string match without an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from simple_agent_lab.messages import (
    TextBlock,
    ToolCallBlock,
    assistant_message,
    tool_result_message,
)
from simple_agent_lab.state import State


@dataclass(frozen=True)
class PlantedFact:
    """A durable fact dropped into the compressible region of a transcript.

    `category` mirrors the SummarizeStrategy prompt's own promise ("durable
    facts, decisions, tool results, constraints, and unresolved questions").
    `needle` is a token unique enough to score retention by substring match.
    """

    category: str
    needle: str
    text: str


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    build: Callable[[], State]
    facts: tuple[PlantedFact, ...]
    # Recent messages a strategy is expected to keep verbatim. The metrics
    # layer feeds this to the strategy under test (keep_recent /
    # keep_recent_exchanges) so "kept tail" stays consistent across scenarios.
    keep_recent: int


# ---------------------------------------------------------------------------
# Scenario 1: a mixed planning dialogue (exercises SummarizeStrategy)
# ---------------------------------------------------------------------------

_DIALOG_FACTS: tuple[PlantedFact, ...] = (
    PlantedFact(
        "fact",
        "rotate-72h",
        "The service API key rotates on a fixed schedule keyed 'rotate-72h'.",
    ),
    PlantedFact(
        "decision",
        "choose-postgres",
        "Decision 'choose-postgres': use PostgreSQL, not MySQL, for the store.",
    ),
    PlantedFact(
        "constraint",
        "py39-floor",
        "Hard constraint 'py39-floor': the code must keep running on Python 3.9.",
    ),
    PlantedFact(
        "tool_result",
        "exit-17",
        "Tool run reported failure 'exit-17' from the integration suite.",
    ),
    PlantedFact(
        "unresolved",
        "cold-cache-miss",
        "Open question 'cold-cache-miss': why the cache misses on cold start.",
    ),
)


def _build_dialog() -> State:
    state = State("Plan the auth-service migration and keep a running memory.")
    state.send("task", "user", "writer", state.task)

    # Older, compressible region: each planted fact lands here, padded so the
    # transcript is comfortably over a few-thousand-token threshold.
    pad = "Context detail. " * 12
    for fact in _DIALOG_FACTS:
        state.send("message", "user", "writer", f"{fact.text} {pad}")
        state.record(
            assistant_message(
                f"Noted: {fact.needle}. Continuing the analysis. {pad}",
                sender="writer",
                target="user",
                kind="step",
            )
        )

    # Recent tail kept verbatim (must be >= keep_recent so the strategy has
    # something to fold *and* something to keep).
    state.send("message", "user", "writer", "What's the immediate next step?")
    state.record(
        assistant_message(
            "Next: wire the rotation hook before the migration.",
            sender="writer",
            target="user",
            kind="step",
        )
    )
    state.send("message", "user", "writer", "Sounds good — proceed.")
    return state


DIALOG = Scenario(
    name="planning-dialogue",
    description=(
        "A multi-turn planning dialogue with five durable facts planted in the "
        "older region. Exercises SummarizeStrategy fact retention."
    ),
    build=_build_dialog,
    facts=_DIALOG_FACTS,
    keep_recent=3,
)


# ---------------------------------------------------------------------------
# Scenario 2: a tool-heavy run (exercises ToolCompactStrategy)
# ---------------------------------------------------------------------------

_TOOL_FACTS: tuple[PlantedFact, ...] = (
    PlantedFact(
        "tool_result", "row-count-4821", "select count(*) => 4821 rows [row-count-4821]"
    ),
    PlantedFact(
        "tool_result",
        "missing-index-idx_user_email",
        "EXPLAIN: missing-index-idx_user_email",
    ),
    PlantedFact("tool_result", "disk-93pct", "df -h => disk-93pct full on /var"),
)


def _build_tool_heavy() -> State:
    state = State("Investigate the slow query and report findings.")
    state.send("task", "user", "writer", state.task)

    commands = [
        ("psql -c 'select count(*)'", _TOOL_FACTS[0].text),
        ("psql -c 'explain analyze ...'", _TOOL_FACTS[1].text),
        ("df -h", _TOOL_FACTS[2].text),
        ("tail -n 50 /var/log/db.log", "log line A\nlog line B\nlog line C"),
        ("uptime", "load average: 0.42 0.51 0.55"),
    ]
    for index, (command, output) in enumerate(commands):
        state.record(
            assistant_message(
                [
                    TextBlock(f"Running step {index}."),
                    ToolCallBlock(f"c{index}", "bash", {"command": command}),
                ],
                sender="writer",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_result_message(
                output,
                tool_call_id=f"c{index}",
                tool_name="bash",
                target="writer",
            )
        )
    return state


TOOL_HEAVY = Scenario(
    name="tool-heavy-investigation",
    description=(
        "Five bash tool exchanges; the first three carry durable findings. "
        "Exercises ToolCompactStrategy folding and preview retention."
    ),
    build=_build_tool_heavy,
    facts=_TOOL_FACTS,
    keep_recent=1,
)


ALL_SCENARIOS: tuple[Scenario, ...] = (DIALOG, TOOL_HEAVY)
