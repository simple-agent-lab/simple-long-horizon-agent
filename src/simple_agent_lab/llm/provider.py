"""Provider configuration.

Provider is *data*, not a class hierarchy. To support a new endpoint,
write an adapter (see `adapters/` and `stream.register_adapter`); to
configure an instance, build a `Provider` literal and pass it around.

A custom Ollama endpoint is just::

    Provider(id="ollama", api="openai-chat",
             base_url="http://localhost:11434/v1",
             model="llama3:8b")

No subclasses, no factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# The set of wire protocols. Each value maps to one adapter in `_ADAPTERS`.
ApiKind = Literal[
    "fake",  # deterministic test adapter
    "anthropic-messages",  # Anthropic Messages API
    "openai-chat",  # OpenAI / OpenAI-compatible Chat Completions
    "openai-responses",  # OpenAI Responses API
]


# Provider-agnostic reasoning depth. A caller says how hard to think *once*;
# each adapter translates this single knob to the field its endpoint expects
# (OpenAI Chat top-level ``reasoning_effort``, OpenAI Responses nested
# ``reasoning={"effort": ...}``, Anthropic ``thinking``/``output_config`` for
# 4.6+ or ``thinking={"budget_tokens": N}`` for older models). The
# model-specific quirks live in the adapters, never in callers. For raw control
# the per-request ``LLMRequest.extra`` passthrough still wins over this.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


@dataclass(frozen=True)
class Provider:
    """Pure-data provider config. JSON-serializable; no callables."""

    id: str  # caller-chosen label, e.g. "claude-prod"
    api: ApiKind
    model: str  # provider's model id, e.g. "claude-sonnet-4-5"
    base_url: str | None = None  # override SDK default (Ollama, Azure, etc.)
    api_key_env: str = ""  # env var name; "" = no key needed (fake / local)
    # `None` means "send no temperature" — the OpenAI Responses API rejects the
    # field, so a responses provider sets this to None and adapters omit it.
    default_temperature: float | None = 1.0
    default_max_tokens: int | None = None
    # Reasoning depth applied when a request doesn't set its own
    # `LLMRequest.reasoning`. Configure once on the provider and every agent
    # run built on it inherits the effort; adapters map it to the wire shape.
    default_reasoning: ReasoningEffort | None = None
    context_window: int | None = None  # advisory only; not enforced
    # When True, adapters that hold the model's prior reasoning replay it
    # on the next request so multi-turn tool-use stays continuous. Flip off
    # only for endpoints that handle reasoning continuity server-side or
    # reject the replayed shape.
    replay_reasoning: bool = True
