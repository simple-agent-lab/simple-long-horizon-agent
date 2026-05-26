"""The streaming protocol and entry points.

Public surface (intentionally small — three functions):
    iter_stream(req)        → Iterator[StreamEvent]   (streaming path)
    complete(req)           → LLMResponse             (blocking helper)
    register_adapter(api, fn)                          (extensibility)

Each adapter is a function `(LLMRequest) -> Iterator[StreamEvent]`.
The last event must always be `kind="done"` carrying an LLMResponse.

The dispatch is keyed off `req.provider.api`. Adapters self-register at
import time (see `adapters/__init__.py`); to add a new provider, write
a function and call `register_adapter("your-api", fn)`.
"""

from __future__ import annotations

from typing import Callable, Iterator

from .types import LLMRequest, LLMResponse, StreamEvent


AdapterFn = Callable[[LLMRequest], Iterator[StreamEvent]]
_ADAPTERS: dict[str, AdapterFn] = {}


def register_adapter(api: str, fn: AdapterFn) -> None:
    """Register a streaming adapter for an api kind.

    Idempotent: re-registering the same api replaces the previous
    adapter (useful for tests and local overrides).
    """
    _ADAPTERS[api] = fn


def iter_stream(req: LLMRequest) -> Iterator[StreamEvent]:
    """Yield events as the LLM produces output.

    Always yields a final `kind="done"` event whose payload contains the
    drained `LLMResponse`. Consumers that only want the final response
    should use `complete()` instead.
    """
    api = req.provider.api
    adapter = _ADAPTERS.get(api)
    if adapter is None:
        raise ValueError(
            f"No adapter registered for api={api!r}. "
            f"Available: {sorted(_ADAPTERS.keys())}"
        )
    yield from adapter(req)


def complete(req: LLMRequest) -> LLMResponse:
    """Block until the LLM finishes; return the drained response.

    Convenience for callers that don't want the streaming events. The
    caller is responsible for any retry / timeout policy beyond what
    the adapter itself does.
    """
    response: LLMResponse | None = None
    for event in iter_stream(req):
        if event.kind == "done":
            response = event.payload["response"]
    if response is None:
        raise RuntimeError(
            f"Stream for api={req.provider.api!r} ended without a 'done' event"
        )
    return response
