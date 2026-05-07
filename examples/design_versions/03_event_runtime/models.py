"""Provider boundary for the event-driven loop.

Single `ModelClient` implementation that delegates to
`simple_agent_lab.llm.complete` for any registered provider. All the
provider-specific logic (Anthropic / OpenAI / fake / future endpoints)
lives behind that one entry point — this file does not import any
provider SDK directly.
"""

from __future__ import annotations

from typing import Any, Optional

from core import (
    Message,
    MetaMode,
    ModelClient,
)
from simple_agent_lab.llm import (
    LLMRequest,
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm import complete as llm_complete
from simple_agent_lab.tools import AgentTool, Tool


class LLMModelClient(ModelClient):
    """`ModelClient` backed by `simple_agent_lab.llm`.

    Construct one per (provider, system_prompt) pair. The client itself
    is stateless across calls; conversation state lives in the agent
    loop's `RuntimeState`.

    `request_extra` is forwarded into `LLMRequest.extra` for request options
    that only a specific provider adapter understands (e.g. `extra_headers`).
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
        request_extra: Optional[dict[str, Any]] = None,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.request_extra = dict(request_extra or {})

    def generate(
        self,
        messages: list[Message],
        meta: MetaMode = "none",
        tools: list[Tool] | None = None,
    ) -> Message:
        req = LLMRequest(
            provider=self.provider,
            messages=messages_to_llm_messages(messages, with_header=meta == "header"),
            tools=[tool_to_llm_tool(t) for t in (tools or [])],
            system_prompt=self.system_prompt or None,
            extra=self.request_extra,
        )
        resp = llm_complete(req)
        return llm_response_to_assistant_message(
            resp,
            sender=self.provider.model,
            target="user" if resp.stop_reason == "end_turn" else "assistant",
            kind="final" if resp.stop_reason == "end_turn" else "thought",
            data={"provider": self.provider.api},
        )


def fake_client(
    model: str = "fake-model",
    system_prompt: str = "",
    request_extra: Optional[dict[str, Any]] = None,
) -> LLMModelClient:
    """Convenience factory: an `LLMModelClient` bound to the fake adapter.

    Equivalent to::

        LLMModelClient(
            Provider(id="fake", api="fake", model=model),
            system_prompt=system_prompt,
            request_extra=request_extra,
        )
    """
    return LLMModelClient(
        LLMProvider(id="fake", api="fake", model=model),
        system_prompt=system_prompt,
        request_extra=request_extra,
    )


def make_client(
    provider_id: str = "fake",
    model: str = "fake-model",
    system_prompt: str = "",
    request_extra: Optional[dict[str, Any]] = None,
) -> LLMModelClient:
    """One-line factory for the common case.

    For real providers supply the full `LLMProvider` via the `provider_id` path
    or construct `LLMModelClient` directly. This keeps the 03 surface small
    while making the happy path obvious.
    """
    if provider_id == "fake":
        return fake_client(model, system_prompt, request_extra)
    # For real providers the caller is expected to build LLMProvider from env
    # or config; we keep the example deterministic by defaulting to fake.
    return fake_client(model, system_prompt, request_extra)
