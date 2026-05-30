"""Model-free metrics over a compression pass.

Everything here is deterministic so it runs in CI: drive a strategy over a
scenario, then score the outcome (did it trigger, how much did it shrink the
context, are tool pairs and pinned kinds still intact, which planted facts
survived into the summary). The live half of the suite reuses these same
scorers on the output of a real LLM compressor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from simple_agent_lab.compression import maybe_compress_context
from simple_agent_lab.context_view import (
    CompressionStrategy,
    ContextPolicy,
    estimate_context_tokens,
)
from simple_agent_lab.core import Agent
from simple_agent_lab.messages import (
    AssistantMessage,
    Message,
    MessageKind,
    assistant_message,
    text_of,
    tool_results_of,
)
from simple_agent_lab.protocols import ContextCompressionEvent
from simple_agent_lab.state import State
from simple_agent_lab.trajectory import json_safe

from .scenarios import PlantedFact, Scenario

EVAL_SCHEMA = "simple-agent-lab.evaluation.v1"


@dataclass(frozen=True)
class EvalResult:
    """Project-owned eval record shape (matches the SWE-bench adapter)."""

    trace_id: str
    scorer: str
    passed: bool
    score: float
    metrics: dict[str, Any]
    reason: str = ""
    meta: dict[str, Any] | None = None


def eval_result_record(result: EvalResult) -> dict[str, Any]:
    return {"schema": EVAL_SCHEMA, "type": "eval_result", **json_safe(result)}


@dataclass(frozen=True)
class CompressionOutcome:
    """What one strategy pass did to one scenario."""

    triggered: bool
    before_tokens: int
    after_tokens: int
    compressed_count: int
    summary_text: str
    active_messages: list[Message] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """after / before; 1.0 when nothing was compressed."""
        if not self.triggered or self.before_tokens == 0:
            return 1.0
        return self.after_tokens / self.before_tokens


def _idle_agent(name: str) -> Agent:
    # maybe_compress_context only reads `agent.name`; generate is never called
    # because we drive compression directly rather than through run().
    return Agent(name, lambda visible: assistant_message("", sender=name))


def run_compression(
    state: State,
    strategy: CompressionStrategy,
    *,
    agent_name: str = "writer",
) -> CompressionOutcome:
    """Apply one strategy to `state` and summarize what changed."""
    before_tokens = estimate_context_tokens(state.active_context_messages())
    policy = ContextPolicy(strategies=(strategy,))
    events = [
        event
        for event in maybe_compress_context(_idle_agent(agent_name), state, policy)
        if isinstance(event, ContextCompressionEvent)
    ]
    active = state.active_context_messages()
    if not events:
        return CompressionOutcome(
            triggered=False,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            compressed_count=0,
            summary_text="",
            active_messages=active,
        )
    last = events[-1]
    summary = state.messages[last.summary_message_index]
    compressed = sum(len(event.compressed_message_indices) for event in events)
    return CompressionOutcome(
        triggered=True,
        before_tokens=events[0].before_tokens,
        after_tokens=last.after_tokens,
        compressed_count=compressed,
        # `text_of`, not `message_text`: the latter is a 120-char one-line
        # preview helper and would silently truncate the summary we score.
        summary_text=text_of(summary.content),
        active_messages=active,
    )


# ---------------------------------------------------------------------------
# Structural scorers
# ---------------------------------------------------------------------------


def tool_pairs_intact(messages: list[Message]) -> bool:
    """Every tool_call in `messages` has its result present, and vice versa.

    A provider rejects an orphaned tool_result and a dangling tool_call has no
    answer to point at, so an intact active view must carry matched id sets.
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            call_ids.update(call.id for call in message.tool_calls)
        result_ids.update(
            block.tool_call_id for block in tool_results_of(message.content)
        )
    return call_ids == result_ids


def pinned_kinds_present(
    before: list[Message],
    after: list[Message],
    kinds: tuple[MessageKind, ...],
) -> bool:
    """No message of a pinned kind was dropped from the active view."""
    before_count = sum(1 for message in before if message.kind in kinds)
    after_count = sum(1 for message in after if message.kind in kinds)
    return after_count >= before_count


def fact_recall(
    summary_text: str,
    facts: tuple[PlantedFact, ...],
) -> tuple[dict[str, bool], float]:
    """Which planted needles survived into the summary, plus the recall rate."""
    hits = {fact.needle: fact.needle in summary_text for fact in facts}
    rate = (sum(hits.values()) / len(hits)) if hits else 1.0
    return hits, rate


# ---------------------------------------------------------------------------
# Threshold behavior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPoint:
    threshold_tokens: int
    triggered: bool
    after_tokens: int


def threshold_trigger_curve(
    scenario: Scenario,
    make_strategy: Callable[[int], CompressionStrategy],
    thresholds: tuple[int, ...],
) -> list[ThresholdPoint]:
    """Sweep `threshold_tokens` and record where compression kicks in.

    A healthy curve is monotonic: high thresholds never trigger, low ones
    always do, and there is a single crossover near the context size.
    """
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        outcome = run_compression(scenario.build(), make_strategy(threshold))
        points.append(
            ThresholdPoint(
                threshold_tokens=threshold,
                triggered=outcome.triggered,
                after_tokens=outcome.after_tokens,
            )
        )
    return points


def curve_is_monotonic(points: list[ThresholdPoint]) -> bool:
    """Once compression stops triggering as threshold rises, it stays off."""
    seen_off = False
    for point in sorted(points, key=lambda p: p.threshold_tokens):
        if not point.triggered:
            seen_off = True
        elif seen_off:
            return False
    return True
