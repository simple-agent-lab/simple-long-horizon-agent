"""Provider adapters.

Importing this package registers the built-in adapters. To add a new
provider, create a sibling module that calls `register_adapter(api, fn)`
at import time, then import it from here (or have the user import it
directly).

The real-provider modules defer their SDK import to the first `stream()`
call, so importing this package never requires `anthropic` or `openai`
to be installed. Calling an adapter without its SDK raises a clear error.
"""

from typing import Any


# Caption emitted in the adjacent user message when a provider's wire shape
# can't carry images inside its tool-result entry (OpenAI Chat / Responses).
# Shared so all OpenAI-shape adapters surface the visual with the same hint.
TOOL_RESULT_VISUAL_CAPTION = "Visual output from {tool_name}:"


def capture_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot a provider request for ``LLMResponse.raw``.

    Returns a shallow copy of the outbound kwargs so every turn's
    ``raw["request"]`` preserves the full request body including the
    messages / input history.
    """
    return dict(kwargs)


def sdk_dump(value: Any) -> Any:
    """Best-effort serialization snapshot of an SDK response object.

    pydantic v2 SDKs (openai, anthropic) expose `model_dump()`. Fall
    back to the raw object when no dump method exists, which is enough
    for `print_trace(raw=True)` to render it via `repr`.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return value


from . import fake  # noqa: F401, E402  (registers "fake")
from . import anthropic_messages  # noqa: F401, E402  (registers "anthropic-messages")
from . import openai_chat  # noqa: F401, E402  (registers "openai-chat")
from . import openai_responses  # noqa: F401, E402  (registers "openai-responses")
