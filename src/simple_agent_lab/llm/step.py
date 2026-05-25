"""LLM-backed `Agent.step` factory.

Bridges the Agent abstraction to the shared `llm/` access layer. Given a
`Provider`, returns a `StepFn` that — on each turn — projects the visible
messages, calls the model, and lifts the response back to an assistant
`Message`.

Kept in `llm/` (not `core/`) because the wiring belongs at the LLM
boundary: it depends on the bridge helpers and on tool/message
projection. `core.py` stays focused on the loop itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .bridge import (
    llm_response_to_assistant_message,
    messages_to_llm_messages,
    tool_to_llm_tool,
)
from .provider import Provider as LLMProvider
from .stream import complete as llm_complete
from .types import LLMRequest

if TYPE_CHECKING:
    from ..core import Agent, State, StepFn
    from ..messages import Message


def make_llm_step(
    provider: LLMProvider,
    *,
    system_prompt: str = "",
    target: str = "all",
    request_extra: Mapping[str, Any] | None = None,
) -> "StepFn":
    """Build an Agent.step function backed by the shared LLM layer."""

    def step(agent: "Agent", visible: "list[Message]", state: "State") -> "Message":
        del state
        request = LLMRequest(
            provider=provider,
            messages=messages_to_llm_messages(visible),
            tools=[tool_to_llm_tool(tool) for tool in agent.tools],
            system_prompt=system_prompt or agent.role or None,
            extra=dict(request_extra or {}),
        )
        response = llm_complete(request)
        kind = "final" if response.stop_reason == "end_turn" else "thought"
        return llm_response_to_assistant_message(
            response,
            sender=agent.name,
            target=target,
            kind=kind,
        )

    return step
