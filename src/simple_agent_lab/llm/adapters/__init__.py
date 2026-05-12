"""Provider adapters.

Importing this package registers the built-in adapters. To add a new
provider, create a sibling module that calls `register_adapter(api, fn)`
at import time, then import it from here (or have the user import it
directly).

The real-provider modules defer their SDK import to the first `stream()`
call, so importing this package never requires `anthropic` or `openai`
to be installed. Calling an adapter without its SDK raises a clear error.
"""

# Caption emitted in the adjacent user message when a provider's wire shape
# can't carry images inside its tool-result entry (OpenAI Chat / Responses).
# Shared so all OpenAI-shape adapters surface the visual with the same hint.
TOOL_RESULT_VISUAL_CAPTION = "Visual output from {tool_name}:"


from . import fake  # noqa: F401, E402  (registers "fake")
from . import anthropic_messages  # noqa: F401, E402  (registers "anthropic-messages")
from . import openai_chat  # noqa: F401, E402  (registers "openai-chat")
from . import openai_responses  # noqa: F401, E402  (registers "openai-responses")
