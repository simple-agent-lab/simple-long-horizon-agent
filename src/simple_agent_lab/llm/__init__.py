"""Unified LLM access layer.

Three-piece public surface:
  - Types: `LLMMessage`, `LLMTool`, `LLMRequest`, `LLMResponse`,
           `ContentBlock`, `ToolCallBlock`, `TokenUsage`, `StreamEvent`.
  - Provider config: `Provider` (data, no subclasses).
  - Calls: `iter_stream(req)` for streaming, `complete(req)` for blocking.

All sync. Each agent loop (01 / 02 / 03) projects its own `Message` and the
shared `simple_agent_lab.tools.Tool` values to these types at the boundary;
this layer never sees agent-loop routing fields (sender, target, kind,
channel).

To add a new provider, write an adapter function and register it::

    from llm import LLMRequest, StreamEvent, register_adapter

    def stream(req: LLMRequest):
        # ... talk to your API, yield StreamEvents, last one kind="done"
        ...

    register_adapter("my-api", stream)
"""

from .provider import ApiKind, Provider
from .bridge import (
    llm_response_to_assistant_message,
    message_to_llm_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .stream import complete, iter_stream, register_adapter
from .types import (
    ContentBlock,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTool,
    MessageContent,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolCallBlock,
    llm_message,
)

# Side-effect: registers the built-in adapters ("fake", and any others
# this package adds in adapters/).
from . import adapters as _adapters  # noqa: F401


__all__ = [
    "ApiKind",
    "ContentBlock",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMTool",
    "MessageContent",
    "Provider",
    "Role",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "ThinkingBlock",
    "TokenUsage",
    "ToolCall",
    "ToolCallBlock",
    "complete",
    "iter_stream",
    "llm_message",
    "llm_response_to_assistant_message",
    "message_to_llm_message",
    "messages_to_llm_messages",
    "register_adapter",
    "tool_to_llm_tool",
]
