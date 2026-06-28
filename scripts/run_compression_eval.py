"""Measure compression *effectiveness*, not just that it runs.

Drives one long tool-heavy scenario through the real `run()` loop under four
policies and prints the effectiveness numbers `summarize_compression` extracts
from the event stream:

    uv run python scripts/run_compression_eval.py

Two things to read off the output:

1. The comparison table — peak/final active tokens, compactions, how much each
   fold removes (kept-fraction), and the retained (append-only) transcript that
   buys recoverability. The model is scripted, so the run is deterministic and
   needs no API key.
2. The scaling check — the same scenario at N and 2N steps. Without compression
   the peak active context roughly doubles (it grows every turn); with a
   strategy the peak barely moves. That bounded peak is the whole point.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (  # noqa: E402
    Agent,
    ContextPolicy,
    Message,
    State,
    SummarizeStrategy,
    TieredStrategy,
    ToolCompactStrategy,
    assistant_message,
    run,
)
from simple_agent_lab.compression import summarize_compression  # noqa: E402
from simple_agent_lab.messages import TextBlock, ToolCallBlock  # noqa: E402
from simple_agent_lab.tools import AgentTool, ToolResult, text_result  # noqa: E402

THRESHOLD = 4000
PolicyFactory = Callable[[], ContextPolicy]


def _summarizer() -> Agent:
    def compressor(visible: list[Message]) -> Message:
        del visible
        return assistant_message(
            "Summary of earlier exploration: files read and key facts retained.",
            sender="compressor",
            target="runtime",
            kind="final",
        )

    return Agent("compressor", compressor)


def _policies() -> dict[str, PolicyFactory]:
    # Fresh per run: strategies may hold a compressor agent / per-run defaults.
    return {
        "none (baseline)": lambda: ContextPolicy(),
        "tool-compact": lambda: ContextPolicy(
            strategy=ToolCompactStrategy(threshold_tokens=THRESHOLD)
        ),
        "summarize": lambda: ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=_summarizer(), threshold_tokens=THRESHOLD, keep_recent=2
            )
        ),
        "tiered": lambda: ContextPolicy(
            strategy=TieredStrategy(
                (
                    ToolCompactStrategy(threshold_tokens=THRESHOLD),
                    SummarizeStrategy(
                        compressor=_summarizer(),
                        threshold_tokens=THRESHOLD,
                        keep_recent=2,
                    ),
                )
            )
        ),
    }


def _read_file(call_id, args, abort, on_update) -> ToolResult:
    del call_id, abort, on_update
    path = args["path"]
    # ~1.5k tokens per observation, so a handful of reads blows past THRESHOLD.
    body = f"# {path}\n" + "\n".join(
        f"line {i:03d}: " + "realistic source content goes here " * 4 for i in range(45)
    )
    return text_result(body)


def _run_tool_heavy(policy: ContextPolicy, n_reads: int) -> list:
    """Agent reads `n_reads` big files in sequence, then finishes."""
    read_tool = AgentTool(
        name="read_file",
        description="Read a source file.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=_read_file,
        execution_mode="sequential",
    )
    turn = {"n": 0}

    def brain(visible: list[Message]) -> Message:
        del visible
        index = turn["n"]
        turn["n"] += 1
        if index < n_reads:
            return assistant_message(
                [
                    TextBlock(f"Reading file {index}."),
                    ToolCallBlock(
                        f"call_{index}", "read_file", {"path": f"f{index}.py"}
                    ),
                ],
                sender="general-purpose",
                target="user",
                kind="step",
            )
        return assistant_message(
            "Done exploring.", sender="general-purpose", target="user", kind="final"
        )

    state = State("Explore the codebase.")
    state.send("task", "user", "general-purpose", state.task)
    agent = Agent("general-purpose", brain, tools=(read_tool,), context_policy=policy)
    return list(run(agent, state, max_turns=n_reads + 2))


def _fmt_by_strategy(folds_by_strategy: dict[str, int]) -> str:
    return (
        ", ".join(f"{name}:{n}" for name, n in sorted(folds_by_strategy.items())) or "-"
    )


def _print_table(n_reads: int) -> dict[str, int]:
    print(f"\n=== tool-heavy scenario, {n_reads} reads (threshold={THRESHOLD}) ===")
    header = (
        f"  {'policy':<16} {'reqs':>5} {'peak_tok':>9} {'kept_frac':>10} "
        f"{'folds_by_strategy':<24}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    peaks: dict[str, int] = {}
    for label, factory in _policies().items():
        metrics = summarize_compression(_run_tool_heavy(factory(), n_reads))
        peaks[label] = metrics.peak_active_tokens
        print(
            f"  {label:<16} {metrics.model_requests:>5} "
            f"{metrics.peak_active_tokens:>9} {metrics.mean_kept_fraction:>10.2f} "
            f"{_fmt_by_strategy(metrics.folds_by_strategy):<24}"
        )
    return peaks


def main() -> None:
    small = _print_table(6)
    large = _print_table(12)

    print("\n=== scaling: peak active tokens, 6 reads -> 12 reads ===")
    print(f"  {'policy':<16} {'peak@6':>8} {'peak@12':>8} {'growth':>8}")
    print("  " + "-" * 44)
    for label in small:
        a, b = small[label], large[label]
        growth = f"{b / a:.2f}x" if a else "n/a"
        print(f"  {label:<16} {a:>8} {b:>8} {growth:>8}")
    print(
        "\nReading it: the baseline peak grows with the number of reads "
        "(unbounded);\ncompression keeps the peak roughly flat (bounded) — "
        "that is the effect.\nThe transcript column stays full either way: "
        "originals are retained for recall."
    )


if __name__ == "__main__":
    main()
