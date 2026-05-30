"""Live-provider plumbing for the compression eval.

Builds the OpenAI-compatible `Provider` from the same env vars the SWE-bench
adapter uses (`OPENAI_MODEL`, `OPENAI_AUTH_TOKEN`, `OPENAI_BASE_URL`,
`API_KIND`) and exposes two live primitives the eval needs:

- `make_compressor_agent` — the `Agent` SummarizeStrategy calls.
- `count_input_tokens` — provider-reported input tokens for a message list,
  the ground truth the char-based estimate is validated against.

Returns `None` from `build_provider_from_env` when the env is not configured
so the runner can skip the live half cleanly instead of crashing.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm.bridge import messages_to_llm_messages
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.llm.stream import complete
from simple_agent_lab.llm.types import LLMMessage, LLMRequest
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import Message

API_KIND_CHOICES = ("openai-chat", "openai-responses")


def build_provider_from_env() -> Provider | None:
    """Build the configured provider, or `None` if the env is incomplete.

    Values are stripped because shells (and some secret stores) leak stray
    whitespace — a leading space on `OPENAI_BASE_URL` otherwise yields an
    "URL missing protocol" connection error.
    """
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    token = (os.environ.get("OPENAI_AUTH_TOKEN") or "").strip()
    if not model or not token:
        return None
    api_kind = (os.environ.get("API_KIND") or "openai-chat").strip()
    if api_kind not in API_KIND_CHOICES:
        api_kind = "openai-chat"
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    return Provider(
        id=api_kind,
        api=api_kind,
        model=model,
        base_url=base_url,
        api_key_env="OPENAI_AUTH_TOKEN",
    )


def request_extra_from_env() -> dict[str, Any]:
    """Optional extra headers some OpenAI-compatible gateways require."""
    session_id = (os.environ.get("OPENAI_SESSION_ID") or "").strip()
    log_id = (os.environ.get("OPENAI_LOG_ID") or "").strip()
    if not session_id or not log_id:
        return {}
    import json

    return {
        "extra_headers": {
            "extra": json.dumps({"session_id": session_id}, separators=(",", ":")),
            "X-TT-logid": log_id,
        }
    }


def make_compressor_agent(
    provider: Provider,
    *,
    name: str = "compressor",
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    return make_llm_agent(
        name=name,
        provider=provider,
        role="You compress conversation context faithfully and concisely.",
        request_extra=dict(request_extra or {}),
    )


def count_input_tokens(
    provider: Provider,
    messages: list[Message],
    *,
    request_extra: Mapping[str, Any] | None = None,
) -> int:
    """Provider-reported input tokens for `messages` + a trivial instruction.

    This is the ground truth the char-based estimate is graded against. The
    trailing instruction is tiny and constant, so its cost is captured by
    the empty-message baseline and can be subtracted by the caller.

    Returns the full input window — non-cached `input_tokens` plus the cached
    portions — so a cache hit doesn't understate the real prompt size (under
    our convention cache counts are additive to `input_tokens`).
    """
    llm_messages = list(messages_to_llm_messages(messages))
    llm_messages.append(LLMMessage(role="user", content="Reply with: ok"))
    request = LLMRequest(
        provider=provider,
        messages=llm_messages,
        extra=dict(request_extra or {}),
    )
    usage = complete(request).usage
    return usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens


# A near-empty probe whose input-token count approximates the provider's fixed
# per-call template overhead (system scaffolding, tool framing, etc.).
BASELINE_PROBE: list[Message] = []


def judge_fact_recall(
    provider: Provider,
    summary_text: str,
    facts: Sequence[tuple[str, str]],
    *,
    request_extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, bool], float]:
    """Semantic fact recall: does the summary *convey* each fact (paraphrase ok)?

    Substring recall undercounts a faithful summary that reworded the fact
    ("4,821 rows" for needle "row-count-4821"). This grades meaning with one
    model call: the model emits one `id: yes|no` line per fact, parsed back
    into a hit map. `facts` is `(id, text)` pairs so this stays decoupled
    from the scenario types.
    """
    if not facts:
        return {}, 1.0
    listing = "\n".join(f"- {fact_id}: {text}" for fact_id, text in facts)
    request = LLMRequest(
        provider=provider,
        system_prompt=(
            "You check whether a summary preserves specific facts. A fact is "
            "PRESENT if the summary conveys its meaning, even reworded or "
            "reformatted; MISSING only if the information is absent. Answer "
            "with exactly one line per fact id in the form `id: yes` or "
            "`id: no`, nothing else."
        ),
        messages=[
            LLMMessage(
                role="user",
                content=f"FACTS:\n{listing}\n\nSUMMARY:\n{summary_text}",
            )
        ],
        extra=dict(request_extra or {}),
    )
    answer = complete(request).text.lower()
    hits = {
        fact_id: any(
            fact_id.lower() in line and "yes" in line for line in answer.splitlines()
        )
        for fact_id, _ in facts
    }
    rate = sum(hits.values()) / len(hits)
    return hits, rate
