"""Model metadata: the per-model rate card and context-window limits.

Both halves are keyed by model id and looked up the same forgiving way, so they
live together: pricing turns `TokenUsage` into dollars, and the window book maps
a model to its context limit.

`messages.TokenUsage` is the per-call *token* primitive; the pricing half is the
*money* layer that sits on top of it. The runtime already snapshots the two
cost primitives — `usage` and `model` — onto every `AssistantMessage`
(`messages.py`) and `ModelResponseEvent` (`protocols.py`), precisely so a
downstream layer can fold cost without re-walking the raw provider blob. This
module is that downstream layer.

Concerns, kept separate:

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
  * `ContextWindowBook` — the context-window lookup: model id -> token limit,
    seeded from a built-in table and overridable from LiteLLM / models.dev
    metadata via the `CONTEXT_WINDOW_BOOK_ENV` file.

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

# Environment override: a path to model metadata with context windows. Accepts
# LiteLLM's `model_prices_and_context_window.json`, models.dev `models.json`,
# and models.dev `api.json` / `catalog.json` provider->models structures.
CONTEXT_WINDOW_BOOK_ENV = "SIMPLE_AGENT_LAB_CONTEXT_WINDOW_BOOK"

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


# Built-in rate card (USD per million tokens), keyed by bare model alias.
# `PriceBook` matches dated snapshots and provider prefixes against these keys,
# so `claude-opus-4-8-20260101`, `anthropic/claude-opus-4-8` and
# `deepseek/deepseek-v4-flash` all resolve here. Entries list only the headline
# `input`/`output` numbers; the cache rates default to the standard Anthropic
# multiples of input (read ~0.1x, 5-minute write ~1.25x) via
# `ModelPrice.from_mapping`, unless a model prices caching differently and lists
# `cache_read`/`cache_write` explicitly (e.g. deepseek's cache read is ~0.02x
# input, and the GLM/OpenAI-compatible models bill no cache write).
_DEFAULT_PRICE_TABLE: dict[str, Mapping[str, float]] = {
    # Anthropic
    "claude-fable-5": {
        "input": 10.0,
        "output": 50.0,
        "cache_read": 1.0,
        "cache_write": 12.5,
    },
    "claude-opus-4-8": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    },
    "claude-opus-4-7": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    },
    "claude-opus-4-6": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    },
    "claude-opus-4-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    },
    "claude-opus-4-1": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.5,
        "cache_write": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    # DeepSeek (cache read ~0.02x input; cache_write unobserved, billed at 0)
    "deepseek-chat": {"input": 0.28, "output": 0.42, "cache_read": 0.028},
    "deepseek-reasoner": {"input": 0.28, "output": 0.42, "cache_read": 0.028},
    "deepseek-v4-flash": {
        "input": 0.14,
        "output": 0.28,
        "cache_read": 0.0028,
        "cache_write": 0.0,
    },
    "deepseek-v4-pro": {
        "input": 0.435,
        "output": 0.87,
        "cache_read": 0.003625,
        "cache_write": 0.0,
    },
    # Zhipu GLM (no cache-write billing)
    "glm-4-32b-0414-128k": {"input": 0.1, "output": 0.1},
    "glm-4.5": {"input": 0.6, "output": 2.2},
    "glm-4.5-air": {"input": 0.2, "output": 1.1},
    "glm-4.5-airx": {"input": 1.1, "output": 4.5},
    "glm-4.5-flash": {"input": 0.0, "output": 0.0},
    "glm-4.5-x": {"input": 2.2, "output": 8.9},
    "glm-4.5v": {"input": 0.6, "output": 1.8},
    "glm-4.6": {"input": 0.6, "output": 2.2, "cache_read": 0.11, "cache_write": 0.0},
    "glm-4.7": {"input": 0.6, "output": 2.2, "cache_read": 0.11, "cache_write": 0.0},
    "glm-5": {"input": 1.0, "output": 3.2, "cache_read": 0.2, "cache_write": 0.0},
    "glm-5-code": {"input": 1.2, "output": 5.0, "cache_read": 0.3, "cache_write": 0.0},
    # OpenAI
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.5},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "cache_read": 0.1},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4, "cache_read": 0.025},
    "gpt-4o": {"input": 2.5, "output": 10.0, "cache_read": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cache_read": 0.075},
    "gpt-5": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
    "gpt-5-chat": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
    "gpt-5-mini": {"input": 0.25, "output": 2.0, "cache_read": 0.025},
    "gpt-5-nano": {"input": 0.05, "output": 0.4, "cache_read": 0.005},
    "gpt-5.1": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
    "gpt-5.1-chat-latest": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
    "gpt-5.2": {"input": 1.75, "output": 14.0, "cache_read": 0.175},
    "gpt-5.2-chat-latest": {"input": 1.75, "output": 14.0, "cache_read": 0.175},
    "gpt-5.3-chat-latest": {"input": 1.75, "output": 14.0, "cache_read": 0.175},
    "gpt-5.4": {"input": 2.5, "output": 15.0, "cache_read": 0.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5, "cache_read": 0.075},
    "gpt-5.4-nano": {"input": 0.2, "output": 1.25, "cache_read": 0.02},
    "gpt-5.5": {"input": 5.0, "output": 30.0, "cache_read": 0.5},
}

DEFAULT_PRICES: dict[str, ModelPrice] = {
    model: ModelPrice.from_mapping(rate) for model, rate in _DEFAULT_PRICE_TABLE.items()
}

DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # Fallbacks for common local aliases when no external window book is provided.
    # Values follow models.dev's provider-agnostic `limit.context` where present.
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "glm-5.2": 1_000_000,
    "z-ai/glm-5.2": 1_000_000,
    "zhipuai/glm-5.2": 1_000_000,
    # OpenAI GPT-5.x. Keys are substrings of the platform deployment ids
    # (e.g. "deployment-gpt-5.4-2026-03-05-platform-global"), so `window_for`'s
    # alias/substring match resolves them.
    "gpt-5.3-codex": 1_000_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.5": 1_000_000,
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


@dataclass(frozen=True)
class ContextWindowBook:
    """Context window lookup keyed by model id."""

    windows: Mapping[str, int]

    def window_for(self, model: str) -> int | None:
        if not model:
            return None
        candidates = _model_lookup_names(model)
        for candidate in candidates:
            exact = self.windows.get(candidate)
            if exact is not None:
                return exact
        for key in sorted(self.windows, key=lambda k: -len(k)):
            aliases = _model_lookup_names(key)
            if any(
                candidate == alias or alias in candidate
                for candidate in candidates
                for alias in aliases
            ):
                return self.windows[key]
        return None

    def with_overrides(self, overrides: Mapping[str, int]) -> "ContextWindowBook":
        merged = dict(self.windows)
        merged.update(overrides)
        return ContextWindowBook(merged)


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


def _load_context_window_overrides() -> dict[str, int]:
    """Read model window metadata from LiteLLM or models.dev JSON."""

    raw = os.environ.get(CONTEXT_WINDOW_BOOK_ENV)
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    if not isinstance(data, dict):
        return out
    for model, spec in data.items():
        if not isinstance(spec, Mapping):
            continue
        _add_window_from_spec(out, str(model), spec)
        models = spec.get("models")
        if isinstance(models, Mapping):
            for child_model, child_spec in models.items():
                if not isinstance(child_spec, Mapping):
                    continue
                model_ids = [str(child_model)]
                if "/" not in str(child_model):
                    model_ids.append(f"{model}/{child_model}")
                for key in ("id", "model", "model_id"):
                    value = child_spec.get(key)
                    if isinstance(value, str):
                        model_ids.append(value)
                        if "/" not in value:
                            model_ids.append(f"{model}/{value}")
                _add_window_from_spec(out, model_ids, child_spec)
    return out


def _add_window_from_spec(
    out: dict[str, int], model: str | list[str], spec: Mapping[str, Any]
) -> None:
    model_ids = [model] if isinstance(model, str) else list(model)
    for key in ("id", "model", "model_id"):
        value = spec.get(key)
        if isinstance(value, str):
            model_ids.append(value)
    window = _window_from_spec(spec)
    if window is None:
        return
    for model_id in model_ids:
        for alias in _model_lookup_names(str(model_id)):
            out[alias] = window


def _window_from_spec(spec: Mapping[str, Any]) -> int | None:
    """Extract a context window from LiteLLM or models.dev style metadata."""

    limit = spec.get("limit")
    if isinstance(limit, Mapping):
        value = limit.get("context") or limit.get("input")
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    for key in ("context", "context_window", "max_input_tokens", "max_tokens"):
        parsed = _positive_int(spec.get(key))
        if parsed is not None:
            return parsed
    return _positive_int(spec.get("max_output_tokens"))


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        raw = value.replace(",", "").replace("_", "").strip()
        if raw.isdigit():
            parsed = int(raw)
            return parsed if parsed > 0 else None
    return None


def _model_lookup_names(model: str) -> tuple[str, ...]:
    """Return exact and common provider-stripped model aliases."""

    exact = model.strip()
    slash_stripped = exact.split("/")[-1]
    names = [exact, slash_stripped]
    # Strip a Bedrock-style provider prefix ("anthropic.claude-opus-4-8" ->
    # "claude-opus-4-8"). Skip it when the dotted suffix is a *version* fragment
    # ("glm-5.2" -> "2"): a numeric alias is meaningless on its own and, as a
    # table key's alias, would substring-match almost any model id in
    # `window_for`'s fallback and return a wrong window for unrelated models.
    dot_stripped = slash_stripped.split(".")[-1]
    if dot_stripped and not dot_stripped[0].isdigit():
        names.append(dot_stripped)
    return tuple(dict.fromkeys(name for name in names if name))


def default_context_window_book() -> ContextWindowBook:
    """The built-in model windows, with external metadata overrides merged in."""

    return ContextWindowBook(DEFAULT_CONTEXT_WINDOWS).with_overrides(
        _load_context_window_overrides()
    )


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
