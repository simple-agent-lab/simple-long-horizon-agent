"""LLM-backed `Agent` factory.

Bundles a `Provider` + agent metadata (`name`, `role`, `tools`) into a
ready-to-run `Agent` whose `generate` callable projects the visible
messages, calls the model, and lifts the response back to an assistant
`Message`.

Lives at the package root (above both `core` and `llm/`) because it
depends on both layers: the runtime `Agent` value from `core` and the
projection helpers / `Provider` plumbing from `llm`. Keeping the
factory out of `llm/` is what lets `llm/` stay free of any agent-loop
concept (no `Agent`, no routing fields, no tool dispatch).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import simple_long_horizon_agent.config as config
from .context_view import ContextPolicy
from .core import Agent, StateInitFn
from .hooks import HookMap
from .llm.bridge import (
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .llm.provider import Provider as LLMProvider
from .llm.provider import ReasoningEffort
from .llm.retry import complete_with_tool_call_retry
from .llm.types import LLMRequest
from .messages import Message
from .tools import AgentTool


def make_llm_agent(
    *,
    name: str,
    provider: LLMProvider,
    role: str = "",
    tools: Sequence[AgentTool] = (),
    system_prompt: str = "",
    target: str = "all",
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
    reasoning: ReasoningEffort | None = None,
    init_state: StateInitFn | None = None,
    hooks: HookMap | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build an `Agent` whose `generate` is backed by `provider`.

    `name`, `role`, and `tools` are needed both by the runtime loop (to
    record events and dispatch tool calls) and by the LLM bridge (to set
    the system prompt and translate tools onto the wire). The factory
    threads them through both layers in one place so callers don't have
    to repeat themselves.

    Two recoverable hiccups are retried here by default
    (`complete_with_tool_call_retry`, which layers `complete_with_retry`):
    transient provider throttling (TPM / rate-limit / 429) and a malformed
    tool call in the model's own output. Every LLM-backed agent gets both
    without each caller re-wrapping `generate`.
    """
    # Resolve the effective system prompt once so the value `generate` sends
    # and the value recorded on the `Agent` (for the request trace) can't drift.
    effective_system_prompt = system_prompt or role or ""
    effective_timeout_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else config.LLM_REQUEST_TIMEOUT.get()
    )

    tools_tuple = tuple(tools)
    llm_tools = [tool_to_llm_tool(tool) for tool in tools_tuple]

    def generate(visible: list[Message]) -> Message:
        request = LLMRequest(
            provider=provider,
            messages=messages_to_llm_messages(visible),
            tools=llm_tools,
            system_prompt=effective_system_prompt or None,
            reasoning=reasoning,
            extra=dict(request_extra or {}),
        )
        # Only override `LLMRequest`'s own default timeout when a caller or env
        # config asked for one.
        if effective_timeout_seconds is not None:
            request = dataclasses.replace(
                request, timeout_seconds=effective_timeout_seconds
            )
        response = complete_with_tool_call_retry(request)
        kind = "final" if response.stop_reason == "end_turn" else "step"
        return llm_response_to_assistant_message(
            response,
            sender=name,
            target=target,
            kind=kind,
        )

    return Agent(
        name=name,
        generate=generate,
        role=role,
        tools=tools_tuple,
        context_policy=context_policy,
        system_prompt=effective_system_prompt,
        llm_provider=provider,
        init_state=init_state,
        hooks=hooks or {},
    )
