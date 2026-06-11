---
title: "Provider-Namespaced `extra` Channel and Two-Layer Trace"
status: Accepted
date: 2026-05-12
slug: extra-channel-and-two-layer-trace
---

# Provider-Namespaced `extra` Channel and Two-Layer Trace

## Status

Accepted

## Context

ADR unify-message-protocol-on-content-blocks unified the message content model on content blocks. Two
provider-facing concerns remained without a principled home:

- **Per-message provider hints.** Anthropic supports per-message prompt-cache
  anchoring via `cache_control` on the last content block; OpenAI Chat takes
  an optional `name` speaker label; Gemini accepts per-message
  `safety_settings`. The original protocol had a one-off `LLMMessage.cache_breakpoint:
  bool` field that no adapter ever read, and `LLMMessage.name` that was set
  by the bridge but also never read. The pattern of "one bespoke field per
  vendor feature" would obviously not scale.
- **What actually went over HTTPS.** The standardized trace shows the
  provider-neutral view (LLMMessage list, content blocks, `model_request` /
  `model_response` events). When debugging "did Anthropic actually see the
  `cache_control` block?" or "did mimo accept the replayed
  `reasoning_content`?" there was no second layer to inspect the verbatim
  wire payload the SDK sent.

The `LLMRequest.extra: dict[str, Any]` field already existed as a
per-call provider-options bag (`seed`, `top_p`, `response_format`, …) that
each adapter whitelisted keys from. We had no symmetric channel at the
per-message scope.

## Decision

Two symmetric `extra` channels, both opt-in and provider-namespaced:

```text
LLMRequest.extra   -> per-call provider options    (seed, top_p, ...)
LLMMessage.extra   -> per-message provider hints   (anthropic.cache_breakpoint,
                                                    openai.name, ...)
```

`LLMMessage.extra: Mapping[str, Any]` is a free-form dict. Keys are
namespaced by provider (`anthropic.cache_breakpoint`,
`openai.name`, `gemini.safety_settings`, …). Each adapter reads only the keys
inside its namespace; unknown keys are **silently ignored** so the same
transcript stays portable across providers.

Runtime agents stash hints under `Message.data["extra"]`; the bridge lifts
them onto `LLMMessage.extra` so they reach the adapter. We do not add an
`extra` field to every runtime `Message` subtype — `data` already exists for
sidecar metadata and the convention "use `data['extra']` for hints" is
enough.

Two-layer trace:

- **Standardized layer** (already existed; ratified here): every
  `AssistantMessage.content` is a `tuple[ContentBlock, ...]`, threaded into
  `model_request` / `model_response` events. Provider-neutral and stable.
- **Raw layer** (new): adapters that make a real SDK call populate
  `LLMResponse.raw: dict[str, Any]` with
  `{"request": <captured kwargs>, "response": <SDK response dump>}`
  at the moment of the call. The request snapshot prunes the
  `messages` / `input` history (it already lives canonical in the
  runtime trajectory, and copying it onto every turn's raw payload
  would turn long sessions into O(N²) memory); everything else
  (model, tools, temperature, system, and outbound `extra`
  translations) is retained so the "did our cache_control or
  reasoning_content land?" debug question remains answerable from
  one turn's snapshot alone. The bridge stashes that raw payload
  under `AssistantMessage.data["raw"]`, so it rides with the message
  through the trajectory. `print_trace(state, raw=True)` renders both
  halves as pretty-printed JSON under each assistant turn.

`raw` serves two needs simultaneously: debugging ("what really
crossed the wire?") and programmatic access to provider-specific
response fields the standardized layer doesn't surface (`refusal`,
`prompt_tokens_details.cached_tokens`, `safety_ratings`, …).

`Provider.replay_reasoning: bool = True` (see ADR unify-message-protocol-on-content-blocks) is the hidden
opt-out for the rare endpoint that rejects replayed reasoning. It lives on
the provider, not on each request, because the decision is endpoint-wide:
once you know mimo accepts replay you set it once and forget it.

The dead `LLMMessage.cache_breakpoint` and `LLMMessage.name` fields are
removed. Their use cases become `extra["anthropic.cache_breakpoint"] = True`
and `extra["openai.name"] = "..."` respectively.

The first wired hint is `anthropic.cache_breakpoint`: when set on a message,
the Anthropic adapter attaches `cache_control: {"type": "ephemeral"}` to the
last wire block of that message — the documented Anthropic anchor pattern.
Same shape works for user, assistant, and tool_result messages.

## Consequences

Adding a new provider-specific message field becomes one adapter change
plus a namespace key choice; no churn in the shared `LLMMessage` type.

A transcript carrying `anthropic.cache_breakpoint` is unchanged when handed
to the OpenAI adapter — the key is outside its namespace, so it is ignored.
Cross-provider portability is preserved without each agent knowing every
target's vocabulary.

Trace becomes legibly two-layered. `print_trace` (default) shows the
standardized view including a dedicated `extra` row per message.
`--raw` turns on the verbatim adapter snapshot alongside, so a reader
can see:

1. the agent's protocol-level hint (`extra: anthropic.cache_breakpoint=True`)
2. the adapter's translation (e.g. `cache_control: {"type": "ephemeral"}`
   appearing on the relevant outbound content block)
3. the model's full response — including fields the standardized layer
   does not surface

without switching tools.

The tradeoff is that `extra` is untyped at the message level. We could
enforce typed dicts per provider, but that would require the caller to know
the provider when constructing the message, defeating the provider-neutral
transcript property. The convention "namespace your keys; adapters
whitelist" is the type story.

`message_text(...)` still falls through to the first ToolResultBlock's
inner text when top-level content has no TextBlock (so tool-result
bundles preview correctly), but otherwise treats `data` as a pure debug
sidecar — `data["raw"]` is for inspection, not for live previews.

## Alternatives Considered

- Typed per-provider `Hints` dataclasses (`AnthropicHints`, `OpenAIHints`).
  Rejected: forces the caller to know which provider will receive the
  message when constructing it, breaking the "same transcript, multiple
  providers" property. The provider-neutral runtime transcript matters more
  than per-field static typing here.
- Loud failure on unknown namespace (raise / log warning). Rejected: an
  agent that sets `extra["anthropic.cache_breakpoint"]` should still be
  routable to OpenAI without code changes. Silent ignore is the right
  default; debugging via `print_trace` or `--raw` shows what happened.
- Keep messages history inside `raw["request"]`. Rejected: it
  duplicates the runtime trajectory and grows the trace memory
  quadratically over a long session. The pruning placeholder
  (`{"_pruned": true, "_count": N}`) keeps the snapshot
  self-describing without the bulk.
- Earlier draft named the field `wire`. Renamed to `raw` because the
  field serves two roles — HTTP-level debug AND programmatic access to
  provider-specific response fields — and "wire" undersells the second
  by sounding like a transport-only concern.
- Per-block `extra` (Anthropic's `cache_control` is technically per-block,
  not per-message). Deferred: the per-message anchor with "applies to last
  block" semantics covers every current use case. Add per-block `extra` to
  `TextBlock` / `ThinkingBlock` / etc. when a real workflow needs
  intra-message cache control.
- Put `raw` on the `model_response` event payload instead of on the
  message. Rejected: the message-attached form rides with the transcript and
  survives event filtering / replay; the event payload is bound to the
  emission moment.
