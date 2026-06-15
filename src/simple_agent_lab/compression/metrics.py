"""Measure compression effectiveness over a run's events.

Correctness ("does a fold happen, are tool pairs kept intact") is covered by
unit tests. This module answers the *effectiveness* questions a strategy is
actually for — borrowed from the agent context-compression survey's D/R/O axes:

- does the active context stay **bounded** instead of growing every turn?
- how much does each fold remove (**density** / compression ratio)?
- what does it **cost** (extra compressor calls, retained transcript)?

It is pure analysis over the event stream `run()` already emits — no new
instrumentation — so it works on any recorded run (live or scripted).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean

from ..protocols import (
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
)


@dataclass(frozen=True)
class CompressionMetrics:
    """Effectiveness summary of one run.

    `peak_active_tokens` / `final_active_tokens` are the model-visible active
    context sized at each model request (`ModelRequestEvent.context_view`); the
    peak is the headline number — a bounded peak across a long run is the whole
    point. `mean_kept_fraction` is the average ``after/before`` over folds (so
    0.25 means each fold cut the active context to a quarter); `tokens_dropped`
    is the summed ``before - after``. `transcript_messages` is the append-only
    total, which never shrinks — the memory cost that buys recoverability (the
    originals stay for `recall`). `folds_by_strategy` attributes each fold to the
    strategy that produced it (`ContextCompressionEvent.strategy`), so a
    mixed/`TieredStrategy` run shows *which* mechanism fired how often — the free
    rule-based fold or the LLM summary — not one undifferentiated count.
    """

    model_requests: int
    compactions: int
    peak_active_tokens: int
    final_active_tokens: int
    tokens_dropped: int
    mean_kept_fraction: float
    transcript_messages: int
    folds_by_strategy: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_requests": self.model_requests,
            "compactions": self.compactions,
            "peak_active_tokens": self.peak_active_tokens,
            "final_active_tokens": self.final_active_tokens,
            "tokens_dropped": self.tokens_dropped,
            "mean_kept_fraction": round(self.mean_kept_fraction, 3),
            "transcript_messages": self.transcript_messages,
            "folds_by_strategy": dict(self.folds_by_strategy),
        }


def summarize_compression(events: Iterable[Event]) -> CompressionMetrics:
    """Reduce a run's events to a `CompressionMetrics`.

    Reads the active-context size off each `ModelRequestEvent` and the
    before/after of each `ContextCompressionEvent`; counts `MessageEvent`s for
    the retained transcript size. A run with no compression yields zero
    compactions and a `mean_kept_fraction` of 1.0 (nothing folded).
    """
    active_series: list[int] = []
    folds: list[tuple[int, int]] = []
    by_strategy: Counter[str] = Counter()
    transcript_messages = 0
    for event in events:
        if isinstance(event, ModelRequestEvent):
            tokens = event.context_view.get("estimated_tokens", 0)
            active_series.append(int(tokens))
        elif isinstance(event, ContextCompressionEvent):
            folds.append((event.before_tokens, event.after_tokens))
            by_strategy[event.strategy or "(unlabeled)"] += 1
        elif isinstance(event, MessageEvent):
            transcript_messages += 1

    kept_fractions = [after / before for before, after in folds if before > 0]
    return CompressionMetrics(
        model_requests=len(active_series),
        compactions=len(folds),
        peak_active_tokens=max(active_series, default=0),
        final_active_tokens=active_series[-1] if active_series else 0,
        tokens_dropped=sum(before - after for before, after in folds),
        mean_kept_fraction=mean(kept_fractions) if kept_fractions else 1.0,
        transcript_messages=transcript_messages,
        folds_by_strategy=dict(by_strategy),
    )
