"""Context compression.

The strategy contract (`CompressionStrategy` / `CompressionDecision`) lives
with `ContextPolicy` in `simple_agent_lab.context_view`; this package holds the
concrete strategies and the runtime that applies them, depending on
`context_view` one-way.

- `strategies` — the strategy-author surface: `ToolCompactStrategy`,
  `SummarizeStrategy`, `DEFAULT_PRESERVE_KINDS`, and the `continuation_preamble`
  that frames a summary fold as a session handoff.
- `agent_control` — agent-controlled compaction: `make_compact_control` pairs
  a `compact` tool (the model requests compression itself) with the strategy
  that applies the request at the next safe point.
- `metrics` — `summarize_compression(events)` measures effectiveness (bounded
  peak, compression ratio, overhead) over a recorded run.
- `runtime` — the framework that turns a `CompressionDecision` into recorded
  events; strategy authors do not need to read it.

A policy holds a single `strategy` (the Strategy pattern); set
`ContextPolicy.strategy` to the one you want and swap implementations to
compare approaches.
"""

from __future__ import annotations

from .agent_control import (
    AgentCompactStrategy,
    CompactControl,
    make_compact_control,
)
from .metrics import CompressionMetrics, summarize_compression
from .runtime import _active_context_tokens, maybe_compress_context
from .strategies import (
    DEFAULT_PRESERVE_KINDS,
    SummarizeStrategy,
    TieredStrategy,
    ToolCompactStrategy,
    continuation_preamble,
    format_index_ranges,
)

__all__ = [
    "AgentCompactStrategy",
    "CompactControl",
    "CompressionMetrics",
    "DEFAULT_PRESERVE_KINDS",
    "SummarizeStrategy",
    "TieredStrategy",
    "ToolCompactStrategy",
    "continuation_preamble",
    "format_index_ranges",
    "make_compact_control",
    "maybe_compress_context",
    "summarize_compression",
    # Re-exported for tests that size an active context directly.
    "_active_context_tokens",
]
