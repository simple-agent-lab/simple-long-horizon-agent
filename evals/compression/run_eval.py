"""Compression eval runner.

Two halves:

- **Offline** (always): threshold-trigger curves, real compression ratios,
  tool-pair safety, and pinned-kind retention for `ToolCompactStrategy` and
  `SummarizeStrategy` (driven by a deterministic fake compressor). No model.

- **Live** (`--live`, needs OPENAI_* env): grades the char-based token estimate
  against provider-reported input tokens (the number `threshold_tokens` is
  really measured in), and grades the SummarizeStrategy prompt by how many
  planted durable facts survive a real summary.

Writes one `eval_result` record per check to
`evals/out/compression/<run-id>/eval_results.jsonl` and prints a summary
table. Run via `bash runs/run_compression_eval.sh [--live]`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.compression import (  # noqa: E402
    DEFAULT_PRESERVE_KINDS,
    SummarizeStrategy,
    ToolCompactStrategy,
)
from simple_agent_lab.context_view import estimate_context_tokens  # noqa: E402
from simple_agent_lab.core import Agent  # noqa: E402
from simple_agent_lab.messages import assistant_message, text_of  # noqa: E402
from simple_agent_lab.trajectory import write_jsonl  # noqa: E402

from evals.compression.metrics import (  # noqa: E402
    EvalResult,
    curve_is_monotonic,
    eval_result_record,
    fact_recall,
    pinned_kinds_present,
    run_compression,
    threshold_trigger_curve,
    tool_pairs_intact,
)
from evals.compression.provider import (  # noqa: E402
    build_provider_from_env,
    count_input_tokens,
    judge_fact_recall,
    make_compressor_agent,
    request_extra_from_env,
)
from evals.compression.scenarios import ALL_SCENARIOS, DIALOG, Scenario  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "evals/out/compression"

# Estimate-vs-real accepted band. `threshold_tokens` is compared against the
# char-based estimate, so an estimate within this factor of real tokens keeps the
# threshold meaningful. Outside it, thresholds need a calibration multiplier.
ESTIMATE_BAND = (0.5, 2.0)
# Live fact-retention bar for the SummarizeStrategy prompt.
FACT_RECALL_FLOOR = 0.8


def _fake_compressor(name: str = "compressor") -> Agent:
    """A deterministic stand-in compressor for offline mechanics checks."""

    def generate(visible):
        joined = " | ".join(text_of(message.content) for message in visible)
        return assistant_message(
            "SUMMARY: " + joined[:300],
            sender=name,
            target="runtime",
            kind="final",
        )

    return Agent(name, generate)


def _adaptive_thresholds(full_tokens: int) -> tuple[int, ...]:
    """Threshold sweep straddling the scenario's full-context size."""
    candidates = {
        0,
        max(1, full_tokens // 4),
        max(1, full_tokens // 2),
        full_tokens,
        full_tokens * 2,
        full_tokens * 100,
    }
    return tuple(sorted(candidates))


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------


def _strategy_for(scenario: Scenario, threshold: int):
    """The compression strategy each scenario is built to exercise."""
    if scenario is DIALOG:
        return SummarizeStrategy(
            compressor=_fake_compressor(),
            threshold_tokens=threshold,
            keep_recent=scenario.keep_recent,
        )
    return ToolCompactStrategy(
        threshold_tokens=threshold,
        keep_recent_exchanges=scenario.keep_recent,
    )


def run_offline(scenarios: Sequence[Scenario]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for scenario in scenarios:
        # Build once: measure `full` and snapshot the pre-compression messages
        # here, then reuse this same state for the compression-pass check below
        # (`threshold_trigger_curve` builds its own fresh states per threshold).
        before = scenario.build()
        before_msgs = before.active_context_messages()
        full = estimate_context_tokens(before_msgs)
        thresholds = _adaptive_thresholds(full)

        # 1. Threshold trigger curve — monotonic with a crossover.
        curve = threshold_trigger_curve(
            scenario, lambda t: _strategy_for(scenario, t), thresholds
        )
        triggered = [p.threshold_tokens for p in curve if p.triggered]
        monotonic = curve_is_monotonic(curve)
        has_crossover = bool(triggered) and len(triggered) < len(curve)
        results.append(
            EvalResult(
                trace_id=f"{scenario.name}/threshold-curve",
                scorer="threshold_trigger_curve",
                passed=monotonic and has_crossover,
                score=1.0 if (monotonic and has_crossover) else 0.0,
                metrics={
                    "full_context_tokens": full,
                    "thresholds": list(thresholds),
                    "triggered_thresholds": triggered,
                    "monotonic": monotonic,
                    "has_crossover": has_crossover,
                },
                reason=(
                    f"compression triggers at thresholds <= ~{max(triggered)} tokens"
                    if triggered
                    else "never triggered across the sweep"
                ),
            )
        )

        # 2. Compression effectiveness + safety at a sub-threshold trigger,
        #    reusing the `before` state built above.
        outcome = run_compression(before, _strategy_for(scenario, max(1, full // 2)))
        intact = tool_pairs_intact(outcome.active_messages)
        pinned = pinned_kinds_present(
            before_msgs, outcome.active_messages, DEFAULT_PRESERVE_KINDS
        )
        shrank = outcome.triggered and outcome.ratio < 1.0
        results.append(
            EvalResult(
                trace_id=f"{scenario.name}/compression-pass",
                scorer="compression_pass",
                passed=shrank and intact and pinned,
                score=round(1.0 - outcome.ratio, 4),
                metrics={
                    "triggered": outcome.triggered,
                    "before_tokens": outcome.before_tokens,
                    "after_tokens": outcome.after_tokens,
                    "ratio": round(outcome.ratio, 4),
                    "compressed_messages": outcome.compressed_count,
                    "tool_pairs_intact": intact,
                    "pinned_kinds_present": pinned,
                },
                reason=(
                    f"shrank {outcome.before_tokens}->{outcome.after_tokens} tokens "
                    f"(ratio {outcome.ratio:.2f}); pairs_intact={intact} pinned={pinned}"
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Live checks
# ---------------------------------------------------------------------------


def run_live(
    scenarios: Sequence[Scenario],
    provider,
    *,
    request_extra,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    overhead = count_input_tokens(provider, [], request_extra=request_extra)

    for scenario in scenarios:
        messages = scenario.build().active_context_messages()
        estimate = estimate_context_tokens(messages)
        real = count_input_tokens(provider, messages, request_extra=request_extra)
        real_content = max(0, real - overhead)
        ratio = (real_content / estimate) if estimate else 0.0
        in_band = ESTIMATE_BAND[0] <= ratio <= ESTIMATE_BAND[1]
        results.append(
            EvalResult(
                trace_id=f"{scenario.name}/token-estimate",
                scorer="token_estimate_accuracy",
                passed=in_band,
                score=round(ratio, 4),
                metrics={
                    "estimated_tokens": estimate,
                    "provider_input_tokens": real,
                    "provider_template_overhead": overhead,
                    "provider_content_tokens": real_content,
                    "real_over_estimate": round(ratio, 4),
                },
                reason=(
                    f"char-based estimate {estimate} vs real {real_content} content "
                    f"tokens (x{ratio:.2f}); +{overhead} fixed template overhead"
                ),
            )
        )

    # SummarizeStrategy prompt fidelity with a real compressor.
    compressor = make_compressor_agent(provider, request_extra=request_extra)
    for scenario in scenarios:
        full = estimate_context_tokens(scenario.build().active_context_messages())
        strategy = SummarizeStrategy(
            compressor=compressor,
            threshold_tokens=max(1, full // 2),
            keep_recent=scenario.keep_recent,
        )
        before = scenario.build()
        before_msgs = before.active_context_messages()
        outcome = run_compression(before, strategy)
        literal_hits, literal_recall = fact_recall(outcome.summary_text, scenario.facts)
        # Semantic recall is the fair grade of the prompt: a faithful summary
        # often rewords a fact, which substring recall would miss.
        judge_hits, judge_recall = judge_fact_recall(
            provider,
            outcome.summary_text,
            [(fact.needle, fact.text) for fact in scenario.facts],
            request_extra=request_extra,
        )
        intact = tool_pairs_intact(outcome.active_messages)
        pinned = pinned_kinds_present(
            before_msgs, outcome.active_messages, DEFAULT_PRESERVE_KINDS
        )
        results.append(
            EvalResult(
                trace_id=f"{scenario.name}/summary-fidelity",
                scorer="summary_fact_recall",
                passed=outcome.triggered
                and judge_recall >= FACT_RECALL_FLOOR
                and intact,
                score=round(judge_recall, 4),
                metrics={
                    "triggered": outcome.triggered,
                    "ratio": round(outcome.ratio, 4),
                    "before_tokens": outcome.before_tokens,
                    "after_tokens": outcome.after_tokens,
                    "semantic_recall": round(judge_recall, 4),
                    "literal_recall": round(literal_recall, 4),
                    "semantic_hits": judge_hits,
                    "literal_hits": literal_hits,
                    "tool_pairs_intact": intact,
                    "pinned_kinds_present": pinned,
                },
                reason=(
                    f"semantic recall {judge_recall:.0%} "
                    f"(literal {literal_recall:.0%}) of {len(scenario.facts)} "
                    f"facts; ratio {outcome.ratio:.2f}"
                ),
                meta={"summary_text": outcome.summary_text},
            )
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: Sequence[EvalResult]) -> None:
    width = max((len(r.trace_id) for r in results), default=10)
    print(f"\n{'CHECK':<{width}}  {'PASS':<5}  SCORE  DETAIL")
    print("-" * (width + 40))
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(
            f"{result.trace_id:<{width}}  {mark:<5}  {result.score:<5}  {result.reason}"
        )
    passed = sum(1 for r in results if r.passed)
    print("-" * (width + 40))
    print(f"{passed}/{len(results)} checks passed\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the compression eval suite.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the live half (token-estimate accuracy + real summary fidelity).",
    )
    parser.add_argument("--run-id", default=None, help="Output subdirectory name.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    results = run_offline(ALL_SCENARIOS)

    if args.live:
        provider = build_provider_from_env()
        if provider is None:
            print(
                "WARNING: --live requested but OPENAI_MODEL/OPENAI_AUTH_TOKEN are "
                "unset; skipping the live half.",
                file=sys.stderr,
            )
        else:
            print(f"Running live checks against model {provider.model!r}...")
            results += run_live(
                ALL_SCENARIOS, provider, request_extra=request_extra_from_env()
            )

    print_summary(results)

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out_dir) / run_id / "eval_results.jsonl"
    write_jsonl(out_path, [eval_result_record(result) for result in results])
    print(f"Wrote {len(results)} eval_result records to {out_path}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
