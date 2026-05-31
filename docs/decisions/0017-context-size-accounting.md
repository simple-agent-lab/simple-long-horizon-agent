# ADR 0017: Context-Size Accounting — Provider Usage, Confined Estimation, a Safety Buffer

## Status

Accepted

## Context

Compression decisions (ADR 0010's context view, the strategies in
`src/simple_agent_lab/compression.py`) need one number: how full is the context
window right now? Getting it wrong is costly in both directions — under-count
and the real window overflows before compression fires; over-count and the
agent compresses too early, burning summarizer calls and shedding context it
still needed.

The runtime has two sources for that number, with very different quality:

- **Provider-reported usage** (`TokenUsage` on an `AssistantMessage`) — the
  exact tokenizer cost of a real call. Ground truth, but only available for
  messages the provider has already counted, and lagging: it describes the
  window *as of the last model response*.
- **A char-based heuristic** (`CHARS_PER_TOKEN`) — available for any message,
  but approximate. A one-off calibration against a real tokenizer measured the
  true ratio varying ~3x by content type (prose ~5.6, code ~3.9, json ~3.1,
  logs ~1.9), so a single char/token constant is structurally unable to be
  accurate.

Two concrete bugs forced the issue. (1) `TokenUsage.context_tokens` summed
`input + output + cache_read + cache_write`, which double-counts on providers
whose `input_tokens` already *includes* the cached portion (OpenAI), inflating
the window estimate ~2x under a warm cache. (2) After a compression dropped
older messages, a kept assistant still carried a pre-compression
`context_tokens`, so the size estimate trusted a baseline that referenced
content no longer present.

## Decision

1. **Provider usage is ground truth; the char estimate is only a fallback.**
   `estimate_context_tokens` takes the latest usage-bearing assistant's
   `context_tokens` as the authoritative size of everything up to it, and uses
   the char heuristic *only* for the tail added since. Estimation is confined
   to the smallest possible slice.

2. **Cache counts are additive to `input_tokens`; adapters normalize to that.**
   The window is `input + output + cache_read + cache_write` only when the four
   are disjoint (Anthropic's shape). OpenAI reports cache as a subset of
   `prompt_tokens`, so its adapters subtract it back out via
   `TokenUsage.from_inclusive_input`. Downstream code never branches on
   provider.

3. **A usage baseline is trusted only if it post-dates the last compression.**
   The compression's `kind="summary"` message is an append-only marker with a
   higher state index than everything it replaced; `_active_context_tokens`
   trusts a baseline only when a usage-bearing assistant was recorded after it.
   The next model turn naturally restores the baseline.

4. **One global, empirically-calibrated char/token default — not per-provider.**
   `CHARS_PER_TOKEN = 3.1` is the measured neutral overall, replacing the old
   `4` guess. Because estimation is confined to the small tail (1), its
   absolute error is bounded and a fixed buffer covers it, so per-provider or
   per-content coefficients are deliberately not worth the machinery.

5. **Compression triggers against an effective window, not the raw one.**
   `effective_token_budget(window) = window − output_reserve − safety_buffer`
   (mirroring opencode's `limit − OUTPUT_TOKEN_MAX − COMPACTION_BUFFER`). The
   buffer is the explicit slack that absorbs the tail estimate's error and any
   not-yet-reported message.

## Consequences

- The size signal is provider-accurate wherever the provider has spoken, and
  only the bounded tail rests on the heuristic — so a fixed buffer, not
  per-model precision, is enough.
- Cache double-counting and stale post-compression baselines are fixed and
  regression-tested (`tests/unit/test_real_adapters.py`,
  `tests/unit/test_core.py`, `tests/unit/test_token_usage.py`).
- The `CompressionStrategy` contract stays `(active, agent_name) -> decision`:
  no per-provider config is threaded into strategies. A caller that knows its
  model passes `threshold_tokens=effective_token_budget(window)` when building
  the policy.
- Weak spot, accepted: a provider that never reports usage estimates the whole
  window, where a fixed buffer is less protective. Real providers report usage;
  the buffer can be widened if one does not.

## Alternatives Considered

- **Pure char estimation for the whole context.** Rejected: error scales with
  context size, so no fixed buffer or single constant can bound it.
- **Per-provider / per-content-type coefficients threaded into strategies.**
  Rejected as over-engineering once estimation was confined to the tail: the
  buffer absorbs the residual, and threading would extend the strategy contract
  for little gain.
- **opencode's pure provider-usage with no tail estimate.** It relies solely on
  the last reported usage plus a buffer, which lags a turn — a large tool output
  can overflow before the next report (opencode issue #10634). Estimating the
  tail lets us react in the same turn; the buffer still backstops the estimate.
- **A visible confirmed/estimated split (`ContextSize`).** Built and then
  removed: with a fixed buffer nothing consumed the estimated fraction, so it
  was speculative observability.
