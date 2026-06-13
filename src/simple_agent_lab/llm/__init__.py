"""Unified LLM access layer.

Three-piece public surface:
  - Types: `LLMMessage`, `LLMTool`, `LLMRequest`, `LLMResponse`,
           `ContentBlock`, `ToolCallBlock`, `TokenUsage`, `StreamEvent`.
  - Provider config: `Provider` (data, no subclasses).
  - Calls: `iter_stream(req)` for streaming, `complete(req)` for blocking.

All sync. The agent runtime projects its own `Message` and the shared
`simple_agent_lab.tools.Tool` values to these types at the boundary; this layer
never sees agent-loop routing fields (sender, target, kind).

To add a new provider, write an adapter function and register it::

    from simple_agent_lab.llm import LLMRequest, StreamEvent, register_adapter

    def stream(req: LLMRequest):
        # ... talk to your API, yield StreamEvents, last one kind="done"
        ...

    register_adapter("my-api", stream)
"""

from .provider import REASONING_EFFORTS, ApiKind, Provider, ReasoningEffort
from .env import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS,
    FAKE_PROVIDER,
    JUDGE_API_KIND_ENV,
    JUDGE_AUTH_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_ENV,
    JUDGE_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
    ProviderEnvNames,
    load_dotenv,
    provider_from_env,
    reasoning_from_env,
    request_extra_from_env,
    resolve_api_key,
)
from .registry import (
    DEFAULT_MODEL_ALIASES,
    FAST_ALIAS,
    STRONG_ALIAS,
    ModelRegistry,
    env_names_for,
)
from .bridge import (
    llm_response_to_assistant_message,
    message_to_llm_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .retry import (
    complete_with_retry,
    complete_with_tool_call_retry,
    invalid_tool_call_reasons,
    is_retryable_llm_error,
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
    ToolCallBlock,
    ToolResultBlock,
    llm_message,
)

# Side-effect: registers the built-in adapters ("fake", and any others
# this package adds in adapters/).
from . import adapters as _adapters  # noqa: F401


__all__ = [
    "ApiKind",
    "REASONING_EFFORTS",
    "ReasoningEffort",
    "API_KIND_CHOICES",
    "API_KIND_ENV",
    "DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS",
    "FAKE_PROVIDER",
    "JUDGE_API_KIND_ENV",
    "JUDGE_AUTH_ENV",
    "JUDGE_BASE_URL_ENV",
    "JUDGE_ENV",
    "JUDGE_MODEL_ENV",
    "OPENAI_AUTH_ENV",
    "OPENAI_BASE_URL_ENV",
    "OPENAI_ENV",
    "OPENAI_LOG_ID_ENV",
    "OPENAI_MODEL_ENV",
    "OPENAI_REASONING_EFFORT_ENV",
    "OPENAI_SESSION_ID_ENV",
    "REASONING_EFFORT_ENV",
    "ProviderEnvNames",
    "ModelRegistry",
    "DEFAULT_MODEL_ALIASES",
    "FAST_ALIAS",
    "STRONG_ALIAS",
    "env_names_for",
    "load_dotenv",
    "provider_from_env",
    "reasoning_from_env",
    "request_extra_from_env",
    "resolve_api_key",
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
    "ToolCallBlock",
    "ToolResultBlock",
    "complete",
    "complete_with_retry",
    "complete_with_tool_call_retry",
    "invalid_tool_call_reasons",
    "is_retryable_llm_error",
    "iter_stream",
    "llm_message",
    "llm_response_to_assistant_message",
    "message_to_llm_message",
    "messages_to_llm_messages",
    "register_adapter",
    "tool_to_llm_tool",
]
