"""Provider adapters.

Importing this package registers the built-in adapters. To add a new
provider, create a sibling module that calls `register_adapter(api, fn)`
at import time, then import it from here (or have the user import it
directly).

The real-provider modules defer their SDK import to the first `stream()`
call, so importing this package never requires `anthropic` or `openai`
to be installed. Calling an adapter without its SDK raises a clear error.
"""

# The shared translation spine. Adapters import from `._spine` directly; these
# names are re-exported here for back-compat with `from .adapters import ...`.
from ._spine import (  # noqa: F401
    TOOL_RESULT_VISUAL_CAPTION,
    capture_request,
    emit_response,
    openai_usage,
    parse_tool_arguments,
    resolve_effort,
    resolve_temperature,
    sdk_dump,
)

from . import fake  # noqa: F401, E402  (registers "fake")
from . import anthropic_messages  # noqa: F401, E402  (registers "anthropic-messages")
from . import openai_chat  # noqa: F401, E402  (registers "openai-chat")
from . import openai_responses  # noqa: F401, E402  (registers "openai-responses")
