"""Provider adapters.

Importing this package registers the built-in adapters. To add a new
provider, create a sibling module that calls `register_adapter(api, fn)`
at import time, then import it from here (or have the user import it
directly).

The real-provider modules defer their SDK import to the first `stream()`
call, so importing this package never requires `anthropic` or `openai`
to be installed. Calling an adapter without its SDK raises a clear error.
"""

from . import fake  # noqa: F401  (registers "fake")
from . import anthropic_messages  # noqa: F401  (registers "anthropic-messages")
from . import openai_chat  # noqa: F401  (registers "openai-chat")
from . import openai_responses  # noqa: F401  (registers "openai-responses")
