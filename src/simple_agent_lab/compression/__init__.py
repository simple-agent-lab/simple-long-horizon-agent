"""Context compression.

The strategy contract (`CompressionStrategy` / `CompressionDecision`) lives
with `ContextPolicy` in `simple_agent_lab.context_view`; this package holds the
concrete strategies and the runtime that applies them, depending on
`context_view` one-way.

- `strategies` — the strategy-author surface: `ToolCompactStrategy`,
  `SummarizeStrategy`, and `DEFAULT_PRESERVE_KINDS`.
- `runtime` — the framework that turns a `CompressionDecision` into recorded
  events; strategy authors do not need to read it.

A policy holds a single `strategy` (the Strategy pattern); set
`ContextPolicy.strategy` to the one you want and swap implementations to
compare approaches.
"""

from __future__ import annotations

from .runtime import _active_context_tokens, maybe_compress_context
from .strategies import (
    DEFAULT_PRESERVE_KINDS,
    SummarizeStrategy,
    TieredStrategy,
    ToolCompactStrategy,
)

__all__ = [
    "DEFAULT_PRESERVE_KINDS",
    "SummarizeStrategy",
    "TieredStrategy",
    "ToolCompactStrategy",
    "maybe_compress_context",
    # Re-exported for tests that size an active context directly.
    "_active_context_tokens",
]
