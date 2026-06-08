"""Pricing: turn `TokenUsage` into dollar cost.

`messages.TokenUsage` is the per-call *token* primitive; this module is the
*money* layer that sits on top of it. The runtime already snapshots the two
cost primitives — `usage` and `model` — onto every `AssistantMessage`
(`messages.py`) and `ModelResponseEvent` (`protocols.py`), precisely so a
downstream layer can fold cost without re-walking the raw provider blob. This
module is that downstream layer.

Three concerns, kept separate:

  * `ModelPrice` / `PriceBook` — the rate card: USD per 1,000,000 tokens, one
    rate per `TokenUsage` bucket (input / output / cache-read / cache-write).
    The cache buckets are first-class, not folded into input, because prompt
    caching prices each differently: a cache *read* is ~0.1x the input rate,
    a 5-minute cache *write* is ~1.25x. Pricing them together would misreport
    the exact cost a caching-heavy agent actually incurs.
  * `usage_cost` — the pure `(usage, price) -> CostBreakdown` function.
  * `RunCost` — the aggregate over a whole run (the main agent plus any
    sub-agent calls reached through tool-result sidecars), with a per-model
    breakdown and a JSON-safe `as_dict()` the trace record embeds.

Everything here is provider-neutral and reads only `messages` (the foundation
layer); the model id is matched against the rate card by `PriceBook`, which
tolerates provider prefixes (`anthropic/…`, `anthropic.…`) and dated
snapshots (`claude-opus-4-8-20260101`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .messages import TokenUsage

# USD rate cards are quoted per million tokens; this is the divisor that turns a
# token count at one of those rates into dollars.
TOKENS_PER_PRICE_UNIT = 1_000_000

# Environment override: a path to a JSON file mapping model id -> rate object
# (`{"input": ..., "output": ..., "cache_read": ..., "cache_write": ...}`),
# merged over the built-in table by `default_price_book()`. Lets an operator
# correct a rate without editing code or re-pricing in a trace meta block.
PRICE_BOOK_ENV = "SIMPLE_AGENT_LAB_PRICE_BOOK"

# Prompt-caching multipliers, relative to the input rate. Anthropic prices a
# cache read at ~0.1x input and a 5-minute-TTL cache write at ~1.25x input
# (a 1-hour write is ~2x — pass `cache_write_ratio=2.0` for that tier).
# `TokenUsage` carries a single `cache_write_tokens` bucket, so a price book
# commits to one write tier; the 5-minute default matches the common path.
CACHE_READ_RATIO = 0.1
CACHE_WRITE_RATIO = 1.25


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, one rate per `TokenUsage` bucket."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPrice":
        """Build from a JSON-style rate object (used by env/file overrides).

        `input` and `output` are required; the cache rates default to the
        standard multiples of `input` when omitted, so an override only needs
        to list the two headline numbers.
        """
        base_input = float(data["input"])
        return cls(
            input=base_input,
            output=float(data["output"]),
            cache_read=float(data.get("cache_read", base_input * CACHE_READ_RATIO)),
            cache_write=float(data.get("cache_write", base_input * CACHE_WRITE_RATIO)),
        )


# Built-in rate card (USD per million tokens), current Anthropic list pricing.
# Cache read is ~0.1x input, 5-minute cache write ~1.25x input. Keys are bare
# model aliases; `PriceBook` matches dated snapshots and provider prefixes
# against them, so `claude-opus-4-8-20260101` and `anthropic/claude-opus-4-8`
# both resolve here.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    #                          input  output  cache_read  cache_write
    "claude-opus-4-8": ModelPrice(5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-7": ModelPrice(5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-6": ModelPrice(5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-5": ModelPrice(5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.3, 3.75),
    "claude-sonnet-4-5": ModelPrice(3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.1, 1.25),
}


@dataclass(frozen=True)
class CostBreakdown:
    """Dollar cost of one or more calls, split by `TokenUsage` bucket."""

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return (
            self.input_usd
            + self.output_usd
            + self.cache_read_usd
            + self.cache_write_usd
        )

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        if not isinstance(other, CostBreakdown):
            return NotImplemented
        return CostBreakdown(
            input_usd=self.input_usd + other.input_usd,
            output_usd=self.output_usd + other.output_usd,
            cache_read_usd=self.cache_read_usd + other.cache_read_usd,
            cache_write_usd=self.cache_write_usd + other.cache_write_usd,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "input_usd": round(self.input_usd, 6),
            "output_usd": round(self.output_usd, 6),
            "cache_read_usd": round(self.cache_read_usd, 6),
            "cache_write_usd": round(self.cache_write_usd, 6),
            "total_usd": round(self.total_usd, 6),
        }


def usage_cost(usage: TokenUsage, price: ModelPrice) -> CostBreakdown:
    """Price one call's `TokenUsage` against a `ModelPrice`.

    The four buckets are priced independently and never folded together — this
    is the single place the cache-vs-fresh-input rate distinction is applied.
    """
    return CostBreakdown(
        input_usd=usage.input_tokens * price.input / TOKENS_PER_PRICE_UNIT,
        output_usd=usage.output_tokens * price.output / TOKENS_PER_PRICE_UNIT,
        cache_read_usd=usage.cache_read_tokens
        * price.cache_read
        / TOKENS_PER_PRICE_UNIT,
        cache_write_usd=usage.cache_write_tokens
        * price.cache_write
        / TOKENS_PER_PRICE_UNIT,
    )


@dataclass(frozen=True)
class PriceBook:
    """A rate card keyed by model id, with forgiving lookup.

    `price_for` strips a provider prefix (`anthropic/…` or `anthropic.…`) and
    then matches the longest rate-card key that is a substring of the model id,
    so dated snapshots (`claude-opus-4-8-20260101`) and Bedrock-style ids
    (`anthropic.claude-opus-4-8`) resolve to the bare alias without a separate
    entry. Returns `None` for an unknown model so callers can report it as
    unpriced rather than silently charging zero.
    """

    prices: Mapping[str, ModelPrice]

    def price_for(self, model: str) -> ModelPrice | None:
        if not model:
            return None
        candidate = model.split("/")[-1].strip()
        exact = self.prices.get(candidate) or self.prices.get(model)
        if exact is not None:
            return exact
        # Longest key wins so "claude-opus-4-8" is preferred over a shorter
        # "claude-opus" style alias if both were ever present.
        for key in sorted(self.prices, key=lambda k: -len(k)):
            if key in candidate:
                return self.prices[key]
        return None

    def with_overrides(self, overrides: Mapping[str, ModelPrice]) -> "PriceBook":
        merged = dict(self.prices)
        merged.update(overrides)
        return PriceBook(merged)


def _load_env_overrides() -> dict[str, ModelPrice]:
    """Read the optional `PRICE_BOOK_ENV` JSON file into rate overrides.

    Best-effort: a missing file is silently ignored (the env var may point at a
    not-yet-created path), but malformed JSON raises so a typo in an operator's
    rate file fails loudly rather than silently mispricing a run.
    """
    raw = os.environ.get(PRICE_BOOK_ENV)
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(model): ModelPrice.from_mapping(rate) for model, rate in data.items()}


def default_price_book() -> PriceBook:
    """The built-in rate card, with any `PRICE_BOOK_ENV` overrides merged in."""
    return PriceBook(DEFAULT_PRICES).with_overrides(_load_env_overrides())


def _sum_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    items = list(usages)
    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in items),
        output_tokens=sum(u.output_tokens for u in items),
        cache_read_tokens=sum(u.cache_read_tokens for u in items),
        cache_write_tokens=sum(u.cache_write_tokens for u in items),
    )


def _usage_dict(tokens: TokenUsage) -> dict[str, int]:
    """The JSON-safe four-bucket shape shared by every cost record."""
    return {
        "input_tokens": tokens.input_tokens,
        "output_tokens": tokens.output_tokens,
        "cache_read_tokens": tokens.cache_read_tokens,
        "cache_write_tokens": tokens.cache_write_tokens,
    }


@dataclass(frozen=True)
class ModelCost:
    """Per-model rollup: how many calls, how many tokens, how many dollars."""

    model: str
    calls: int
    tokens: TokenUsage
    cost: CostBreakdown

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "tokens": _usage_dict(self.tokens),
            "cost": self.cost.as_dict(),
        }


@dataclass(frozen=True)
class RunCost:
    """Aggregate cost of a run, broken down per model.

    `by_model` is sorted most-expensive first. `unpriced_models` lists model
    ids that reported usage but had no rate-card entry — their tokens are
    counted but contribute $0, so a caller can flag "this number is a lower
    bound" rather than trusting a silent zero.
    """

    by_model: tuple[ModelCost, ...] = ()
    unpriced_models: tuple[str, ...] = ()

    @property
    def total_cost(self) -> CostBreakdown:
        total = CostBreakdown()
        for entry in self.by_model:
            total = total + entry.cost
        return total

    @property
    def total_usd(self) -> float:
        return self.total_cost.total_usd

    @property
    def total_tokens(self) -> TokenUsage:
        return _sum_usage(entry.tokens for entry in self.by_model)

    @property
    def calls(self) -> int:
        return sum(entry.calls for entry in self.by_model)

    def as_dict(self) -> dict[str, Any]:
        # Fold the per-model entries once, not once per aggregate property.
        total_cost = self.total_cost
        return {
            "total_usd": round(total_cost.total_usd, 6),
            "calls": self.calls,
            "total_tokens": _usage_dict(self.total_tokens),
            "cost": total_cost.as_dict(),
            "by_model": [entry.as_dict() for entry in self.by_model],
            "unpriced_models": list(self.unpriced_models),
        }

    @classmethod
    def from_calls(
        cls,
        calls: Iterable[tuple[str, TokenUsage]],
        price_book: PriceBook | None = None,
    ) -> "RunCost":
        """Fold an iterable of `(model, usage)` pairs into a per-model rollup."""
        book = price_book or default_price_book()
        grouped: dict[str, list[TokenUsage]] = {}
        for model, usage in calls:
            grouped.setdefault(model, []).append(usage)

        entries: list[ModelCost] = []
        unpriced: list[str] = []
        for model, usages in grouped.items():
            tokens = _sum_usage(usages)
            price = book.price_for(model)
            cost = usage_cost(tokens, price) if price is not None else CostBreakdown()
            if price is None:
                unpriced.append(model)
            entries.append(
                ModelCost(
                    model=model or "(unknown)",
                    calls=len(usages),
                    tokens=tokens,
                    cost=cost,
                )
            )
        entries.sort(key=lambda e: e.cost.total_usd, reverse=True)
        return cls(by_model=tuple(entries), unpriced_models=tuple(sorted(unpriced)))

    @classmethod
    def from_run(
        cls,
        events: Iterable[Any],
        messages: Iterable[Any] = (),
        price_book: PriceBook | None = None,
    ) -> "RunCost":
        """Fold a run's model calls — main agent plus sub-agents — into a rollup.

        Reads `model_response` events (which carry the `usage`/`model` cost
        primitives) for the main agent, then descends into any sub-agent event
        logs stashed on tool-result message sidecars by `task_tool`, recursively.
        Accepts both live runtime objects and the JSON-safe dicts a persisted
        trace deserializes to, so it prices a `State` and a `RunTrace` the same
        way.
        """
        return cls.from_calls(_iter_run_calls(events, messages), price_book)


# --- duck-typed extraction over live objects *and* JSON-safe dicts ----------
#
# The same fold runs against a live `State` (events are dataclass instances,
# usage is a `TokenUsage`) and a deserialized `RunTrace` (events are dicts,
# usage is a dict). These helpers normalize both shapes so the logic above
# stays single-path.


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _kind_str(event: Any) -> str:
    kind = _attr(event, "kind")
    # Live events carry an `EventKind` enum; its `.value` is the wire string.
    return str(getattr(kind, "value", kind) or "")


def _coerce_usage(value: Any) -> TokenUsage | None:
    if value is None:
        return None
    if isinstance(value, TokenUsage):
        usage = value
    elif isinstance(value, Mapping):
        usage = TokenUsage(
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            cache_read_tokens=int(value.get("cache_read_tokens", 0)),
            cache_write_tokens=int(value.get("cache_write_tokens", 0)),
        )
    else:
        return None
    # All-zero usage is the runtime's "unknown", not a real $0 call — skip it
    # so it doesn't inflate the call count with phantom turns.
    return usage if usage.context_tokens > 0 else None


def _sub_event_lists(message: Any) -> Iterable[list[Any]]:
    """Yield each sub-agent event log stashed on a tool-result message.

    `task_tool` records `details={"sub_events": list(state.events)}` per
    sub-agent call; the sidecar keys those by tool-call id under `details`.
    """
    sidecar = _attr(message, "sidecar")
    if not isinstance(sidecar, Mapping):
        return
    details = sidecar.get("details")
    if not isinstance(details, Mapping):
        return
    for call_details in details.values():
        if not isinstance(call_details, Mapping):
            continue
        sub_events = call_details.get("sub_events")
        if isinstance(sub_events, list) and sub_events:
            yield sub_events


def _messages_in_events(events: Iterable[Any]) -> list[Any]:
    """Pull the runtime messages back out of an event log.

    Sub-agent events are stored as a flat event list, but a sub-sub-agent's
    events live on a tool-result *message* inside that list (a `message`
    event). Recovering those messages lets the fold recurse to any depth.
    """
    messages: list[Any] = []
    for event in events:
        if _kind_str(event) == "message":
            message = _attr(event, "message")
            if message is not None:
                messages.append(message)
    return messages


def _iter_run_calls(
    events: Iterable[Any],
    messages: Iterable[Any],
) -> Iterable[tuple[str, TokenUsage]]:
    for event in events:
        if _kind_str(event) != "model_response":
            continue
        usage = _coerce_usage(_attr(event, "usage"))
        if usage is None:
            continue
        yield str(_attr(event, "model") or ""), usage

    for message in messages:
        for sub_events in _sub_event_lists(message):
            yield from _iter_run_calls(sub_events, _messages_in_events(sub_events))


__all__ = [
    "CACHE_READ_RATIO",
    "CACHE_WRITE_RATIO",
    "CostBreakdown",
    "DEFAULT_PRICES",
    "ModelCost",
    "ModelPrice",
    "PRICE_BOOK_ENV",
    "PriceBook",
    "RunCost",
    "TOKENS_PER_PRICE_UNIT",
    "default_price_book",
    "usage_cost",
]
