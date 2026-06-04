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

Per-turn model switching lives here too (see `ProviderSelector`): the
agent can use a different model on each turn without touching the core
loop or the `generate(visible) -> Message` contract. The factory just
resolves which provider to call before each model step.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .context_view import ContextPolicy
from .core import Agent
from .llm.bridge import (
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .llm.provider import Provider as LLMProvider
from .llm.retry import complete_with_tool_call_retry
from .llm.types import LLMRequest
from .messages import AssistantMessage, Message
from .tools import AgentTool

# A `ProviderSelector` picks the provider for one turn from the zero-based turn
# index (turn 0 is the agent's first model step). It is `Provider` as *data*
# turned into a per-turn decision: return a different `Provider` per turn to
# switch models each round, or the same one to pin a model. A plain list is just
# `lambda turn: models[turn % len(models)]` (cycle) or
# `lambda turn: models[min(turn, len(models) - 1)]` (escalate-then-hold); the
# factory keeps only the callable form so the policy stays in caller code rather
# than a hidden list-exhaustion rule.
ProviderSelector = Callable[[int], LLMProvider]

# `provider` accepts a single `Provider` (one model for the whole run, the
# common case) or a `ProviderSelector` (a model per turn).
ProviderLike = LLMProvider | ProviderSelector


def _provider_for_turn(provider: ProviderLike, turn: int) -> LLMProvider:
    """Resolve the concrete `Provider` for `turn`.

    A bare `Provider` is used as-is on every turn; anything else is treated
    as a `ProviderSelector` and called with the turn index.
    """
    if isinstance(provider, LLMProvider):
        return provider
    return provider(turn)


def _turn_index(visible: list[Message], name: str) -> int:
    """The zero-based turn this agent is about to take.

    Derived statelessly from the visible context: the count of assistant
    messages this agent has already produced. The loop calls `generate`
    once per turn and records the output before the next turn, so this is
    0 on the first step, 1 on the second, and so on — and it resets on its
    own for each fresh run (no mutable counter to leak across `agent.run`
    calls). If compression folds away some of the agent's own earlier
    messages the count can dip; per-turn routing is a model-choice policy,
    so an occasional repeated choice after compaction is harmless.
    """
    return sum(
        1
        for message in visible
        if isinstance(message, AssistantMessage) and message.sender == name
    )


def make_llm_agent(
    *,
    name: str,
    provider: ProviderLike,
    role: str = "",
    tools: Sequence[AgentTool] = (),
    system_prompt: str = "",
    target: str = "all",
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build an `Agent` whose `generate` is backed by `provider`.

    `name`, `role`, and `tools` are needed both by the runtime loop (to
    record events and dispatch tool calls) and by the LLM bridge (to set
    the system prompt and translate tools onto the wire). The factory
    threads them through both layers in one place so callers don't have
    to repeat themselves.

    `provider` is either a single `Provider` (one model for the whole run)
    or a `ProviderSelector` — a `(turn) -> Provider` callable resolved
    before each model step, so the agent can switch models every round
    (cheap model to explore, strong model to finish, alternate per turn,
    etc.). Per-turn switching needs no change to the core loop or the
    `generate(visible) -> Message` contract; the served model lands on each
    response (`AssistantMessage.model` / `ModelResponseEvent.model`) so a
    trace shows which model answered each turn.

    Two recoverable hiccups are retried here by default
    (`complete_with_tool_call_retry`, which layers `complete_with_retry`):
    transient provider throttling (TPM / rate-limit / 429) and a malformed
    tool call in the model's own output. Every LLM-backed agent gets both
    without each caller re-wrapping `generate`.
    """
    tools_tuple = tuple(tools)
    # Resolve the effective system prompt once so the value `generate` sends
    # and the value recorded on the `Agent` (for the request trace) can't drift.
    effective_system_prompt = system_prompt or role or ""

    def generate(visible: list[Message]) -> Message:
        turn_provider = _provider_for_turn(provider, _turn_index(visible, name))
        request = LLMRequest(
            provider=turn_provider,
            messages=messages_to_llm_messages(visible),
            tools=[tool_to_llm_tool(tool) for tool in tools_tuple],
            system_prompt=effective_system_prompt or None,
            extra=dict(request_extra or {}),
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
    )
