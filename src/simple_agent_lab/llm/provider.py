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
from typing import Literal, Optional


# The set of wire protocols. Each value maps to one adapter in `_ADAPTERS`.
ApiKind = Literal[
    "fake",                  # deterministic test adapter
    "anthropic-messages",    # Anthropic Messages API
    "openai-chat",           # OpenAI / OpenAI-compatible Chat Completions
    "openai-responses",      # OpenAI Responses API
]


@dataclass(frozen=True)
class Provider:
    """Pure-data provider config. JSON-serializable; no callables."""
    id: str                              # caller-chosen label, e.g. "claude-prod"
    api: ApiKind
    model: str                           # provider's model id, e.g. "claude-sonnet-4-5"
    base_url: Optional[str] = None       # override SDK default (Ollama, Azure, etc.)
    api_key_env: str = ""                # env var name; "" = no key needed (fake / local)
    default_temperature: float = 1.0
    default_max_tokens: Optional[int] = None
    context_window: Optional[int] = None  # advisory only; not enforced
