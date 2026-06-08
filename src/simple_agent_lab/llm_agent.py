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

Per-round model switching lives here too: give `make_llm_agent` a map of
named models plus a `choose_model(ctx) -> name` function, and the factory
calls the chosen model on each round (see `make_llm_agent`). It needs no
change to the core loop or the `generate(visible) -> Message` contract; the
factory just picks which model to call before each model step.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .messages import AssistantMessage, Message, tool_results_of
from .tools import AgentTool


@dataclass(frozen=True)
class RoundContext:
    """What a `choose_model` function sees before one model step.

    `round` is the zero-based round index — 0 on the agent's first step, 1
    on the next, and so on. `messages` is the visible context for this round
    (the same list `generate` receives), so a chooser can inspect anything it
    needs. `last_failed` is the curated common signal: did the most recent
    tool result in view come back as an error? It lets a chooser escalate
    after a bad round, e.g. ``"strong" if ctx.last_failed else "fast"``.
    """

    round: int
    messages: tuple[Message, ...]

    @property
    def last_failed(self) -> bool:
        for message in reversed(self.messages):
            results = tool_results_of(message.content)
            if results:
                return any(block.is_error for block in results)
        return False


# `choose_model` maps the current round's context to a key in the model map.
ModelChooser = Callable[[RoundContext], str]

# `provider` accepts a single `Provider` (one model for the whole run, the
# common case) or a map of named models (`{"fast": ..., "strong": ...}`). With
# a map, `choose_model` names which model to use each round.
ProviderLike = LLMProvider | Mapping[str, LLMProvider]


def _provider_for_round(
    provider: ProviderLike,
    choose_model: ModelChooser | None,
    visible: list[Message],
    name: str,
) -> LLMProvider:
    """Resolve the `Provider` to call for the round the agent is about to take.

    A bare `Provider` is used as-is (the round scan and chooser are skipped).
    For a model map, `choose_model` is called with the round's `RoundContext`
    and must return a key in the map.

    The round index is derived statelessly from the visible context — the count
    of assistant messages this agent has already produced — so it resets on its
    own for each fresh `agent.run(...)` with no mutable counter to leak across
    runs. (If compression folds away some of the agent's own earlier messages
    the count can dip; per-round routing is a model-choice policy, so a repeated
    choice is harmless.)
    """
    if isinstance(provider, LLMProvider):
        return provider
    assert choose_model is not None  # guaranteed by make_llm_agent's validation
    round_index = sum(
        1
        for message in visible
        if isinstance(message, AssistantMessage) and message.sender == name
    )
    context = RoundContext(round=round_index, messages=tuple(visible))
    key = choose_model(context)
    try:
        return provider[key]
    except KeyError:
        raise KeyError(
            f"choose_model returned {key!r}, which is not in the model map "
            f"{sorted(provider)}"
        ) from None


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
    choose_model: ModelChooser | None = None,
) -> Agent:
    """Build an `Agent` whose `generate` is backed by `provider`.

    `name`, `role`, and `tools` are needed both by the runtime loop (to
    record events and dispatch tool calls) and by the LLM bridge (to set
    the system prompt and translate tools onto the wire). The factory
    threads them through both layers in one place so callers don't have
    to repeat themselves.

    `provider` is either a single `Provider` (one model for the whole run)
    or a **map of named models** — `{"fast": ..., "strong": ...}` — paired
    with `choose_model`, a `(RoundContext) -> name` function the factory calls
    before each round to pick which model from the map serves that round::

        make_llm_agent(
            name="agent",
            provider={"fast": fast, "strong": strong},
            choose_model=lambda ctx: "fast" if ctx.round == 0 else "strong",
        )

    The chooser can route on round number or on conversation state — e.g.
    ``"strong" if ctx.last_failed else "fast"`` to escalate after a failed
    tool call. Per-round switching needs no change to the core loop or the
    `generate(visible) -> Message` contract; the served model lands on each
    response (`AssistantMessage.model` / `ModelResponseEvent.model`) so a
    trace shows which model answered each round.

    Two recoverable hiccups are retried here by default
    (`complete_with_tool_call_retry`, which layers `complete_with_retry`):
    transient provider throttling (TPM / rate-limit / 429) and a malformed
    tool call in the model's own output. Every LLM-backed agent gets both
    without each caller re-wrapping `generate`.
    """
    if isinstance(provider, LLMProvider):
        if choose_model is not None:
            raise ValueError(
                "choose_model only applies to a model map; a single Provider "
                "needs no chooser"
            )
    elif not provider:
        raise ValueError("model map must hold at least one Provider")
    elif choose_model is None:
        raise ValueError(
            "a model map needs choose_model to name which model serves each round"
        )

    tools_tuple = tuple(tools)
    # Resolve the effective system prompt once so the value `generate` sends
    # and the value recorded on the `Agent` (for the request trace) can't drift.
    effective_system_prompt = system_prompt or role or ""

    def generate(visible: list[Message]) -> Message:
        request = LLMRequest(
            provider=_provider_for_round(provider, choose_model, visible, name),
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
