"""Default retries for the LLM-backed `generate`.

A long agent run makes many provider calls; two recoverable hiccups should not
abort it. Both retries are provider-layer concerns — they talk only
`LLMRequest` / `LLMResponse`, no agent-loop fields — so they live in `llm/` and
are applied once, by default, where the LLM-backed `generate` is built
(`make_llm_agent`). Callers do not re-wrap them per run.

1. `complete_with_retry` — transient *provider* throttling (TPM / rate-limit /
   429). Retries only errors that look like throttling (a real bad-request /
   auth / schema error still surfaces immediately) with capped exponential
   backoff.
2. `complete_with_tool_call_retry` — a malformed tool call in the model's *own
   output* (a call to a tool that isn't on offer, or arguments that weren't
   valid JSON). Re-asks the model with a short corrective note appended, so a
   plain re-roll at temperature 0 doesn't just repeat the same bad output.

`complete_with_tool_call_retry` is the composite the agent uses: its
`complete_fn` defaults to `complete_with_retry`, so one call covers both.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Callable

from .stream import complete
from .types import (
    RAW_ARGUMENTS_KEY,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTool,
    llm_message,
)

RETRY_MAX_ATTEMPTS = 20
RETRY_INITIAL_DELAY_S = 4.0
RETRY_MAX_DELAY_S = 60.0

# Total model calls allowed when repairing an invalid tool call: the first call
# plus up to (N - 1) re-asks. Small on purpose — if the model can't produce a
# valid call in a few tries we degrade to the last response and let the agent
# loop surface the tool error so the model self-corrects on the next turn.
TOOL_CALL_RETRY_MAX_ATTEMPTS = 3


def is_retryable_llm_error(exc: BaseException) -> bool:
    """True for transient provider throttling (TPM / rate-limit / 429)."""

    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "tpm",
            "tokens per minute",
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
        )
    )


def complete_with_retry(
    request: LLMRequest,
    *,
    complete_fn: Callable[[LLMRequest], LLMResponse] = complete,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    initial_delay_s: float = RETRY_INITIAL_DELAY_S,
    max_delay_s: float = RETRY_MAX_DELAY_S,
    sleep_fn: Callable[[float], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> LLMResponse:
    """Call `complete_fn(request)`, retrying transient throttling with backoff.

    `complete_fn` defaults to the blocking `complete`; tests inject a fake.
    Non-throttling errors are re-raised on the first attempt. After
    `max_attempts` the last throttling error propagates. `sleep_fn` defaults
    to `time.sleep`, resolved at call time so it stays patchable.
    """

    sleep = sleep_fn if sleep_fn is not None else time.sleep
    delay = initial_delay_s
    for attempt in range(1, max_attempts + 1):
        try:
            return complete_fn(request)
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_llm_error(exc):
                raise
            _log(
                log_fn,
                f"LLM retryable error (attempt {attempt}/{max_attempts}); "
                f"retrying in {delay:g}s: {type(exc).__name__}: {exc}",
            )
            sleep(delay)
            delay = min(delay * 2, max_delay_s)
    raise RuntimeError("unreachable LLM retry state")


def invalid_tool_call_reasons(
    response: LLMResponse, tools: Sequence[LLMTool]
) -> list[str]:
    """Why each tool call in `response` is malformed; ``[]`` means all valid.

    Catches the two structural slips a model can make: calling a tool that
    isn't on offer, and emitting arguments the provider couldn't parse as JSON
    (adapters surface that as the ``RAW_ARGUMENTS_KEY`` sentinel). Schema-level
    gaps like a missing required field are left to the tool's own error result,
    which the agent loop already feeds back to the model.
    """

    known = {tool.name for tool in tools}
    reasons: list[str] = []
    for call in response.tool_calls:
        if call.name not in known:
            reasons.append(f"unknown tool {call.name!r}")
        elif RAW_ARGUMENTS_KEY in call.arguments:
            reasons.append(f"tool {call.name!r} had arguments that were not valid JSON")
    return reasons


def complete_with_tool_call_retry(
    request: LLMRequest,
    *,
    complete_fn: Callable[[LLMRequest], LLMResponse] = complete_with_retry,
    max_attempts: int = TOOL_CALL_RETRY_MAX_ATTEMPTS,
    log_fn: Callable[[str], None] | None = None,
) -> LLMResponse:
    """Re-ask the model when it emits a structurally invalid tool call.

    `complete_fn` defaults to `complete_with_retry`, so throttling is handled
    underneath. A malformed tool call is the model's own output, so each re-ask
    appends a corrective note to the request (changing the input so a
    temperature-0 model doesn't just repeat itself). After `max_attempts` the
    last response is returned as-is — an unrepairable call then degrades through
    the normal tool-error path instead of aborting the run.
    """

    response = complete_fn(request)
    reasons = invalid_tool_call_reasons(response, request.tools)
    for attempt in range(1, max_attempts):
        if not reasons:
            return response
        _log(
            log_fn,
            f"invalid tool call (re-ask {attempt}/{max_attempts - 1}): "
            f"{'; '.join(reasons)}",
        )
        repaired = replace(
            request,
            messages=[
                *request.messages,
                _tool_call_repair_message(reasons, request.tools),
            ],
        )
        response = complete_fn(repaired)
        reasons = invalid_tool_call_reasons(response, request.tools)
    return response


def _tool_call_repair_message(
    reasons: Sequence[str], tools: Sequence[LLMTool]
) -> LLMMessage:
    """A `system`-role nudge listing what was wrong and which tools are valid.

    `system` role (not `user`) so it is wire-portable: adapters fold it into the
    provider's system field / accept it inline, side-stepping the strict
    user/assistant alternation Anthropic enforces.
    """

    names = ", ".join(sorted(tool.name for tool in tools))
    available = (
        f"Only call these tools: {names}." if names else "No tools are available."
    )
    return llm_message(
        "system",
        "A tool call in your previous reply was invalid "
        f"({'; '.join(reasons)}). {available} "
        "Reply again using valid tool calls with correct JSON arguments.",
    )


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)
        return
    print(message, file=sys.stderr, flush=True)
