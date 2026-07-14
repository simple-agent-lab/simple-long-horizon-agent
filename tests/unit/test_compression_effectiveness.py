"""Effectiveness of the compression strategies, not just that they run.

These assert the load-bearing *effect* — borrowed from the agent
context-compression survey's D/R axes — over deterministic scripted runs:

- the active context stays **bounded** under compression while it grows
  unbounded without it (the whole point of compaction);
- each fold actually **reduces** the active context (density);
- the task and the recent tail are never folded away;
- a fact folded out for density is still **recoverable** through `recall`
  (compression is lossless externalization, not deletion).

`tests/unit/test_compression_control.py` covers the mechanics (a fold happens,
tool pairs stay intact); this file measures whether it works.
"""

from __future__ import annotations

import unittest

from simple_agent_lab import (
    Agent,
    ContextCompressionEvent,
    ContextPolicy,
    Message,
    State,
    SummarizeStrategy,
    TieredStrategy,
    ToolCompactStrategy,
    assistant_message,
    run,
    text_of,
)
from simple_agent_lab.compression import maybe_compress_context, summarize_compression
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import TextBlock, ToolCallBlock
from simple_agent_lab.messages import (
    ToolResultBlock,
    message_tool_calls,
    tool_results_message,
    tool_results_of,
    user_message,
)
from simple_agent_lab.tools import make_recall_tool, tool_result_text

THRESHOLD = 4000
REAL_PROVIDER = Provider(id="test", api="openai-chat", model="test-model")


def _no_abort() -> bool:
    return False


def _read_tool():
    from simple_agent_lab.tools import AgentTool, text_result

    def execute(call_id, args, abort, on_update):
        del call_id, abort, on_update
        body = f"# {args['path']}\n" + "\n".join(
            f"line {i}: " + "content " * 8 for i in range(45)
        )
        return text_result(body)

    return AgentTool(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
        execution_mode="sequential",
    )


def _run_tool_heavy(policy: ContextPolicy, n_reads: int) -> list:
    """Read `n_reads` big files in sequence under `policy`, then finish."""
    turn = {"n": 0}

    def brain(visible: list[Message]) -> Message:
        del visible
        index = turn["n"]
        turn["n"] += 1
        if index < n_reads:
            return assistant_message(
                [
                    TextBlock(f"read {index}"),
                    ToolCallBlock(f"c{index}", "read_file", {"path": f"f{index}.py"}),
                ],
                sender="general-purpose",
                target="user",
                kind="step",
            )
        return assistant_message(
            "done", sender="general-purpose", target="user", kind="final"
        )

    state = State("explore")
    state.send("task", "user", "general-purpose", state.task)
    agent = Agent(
        "general-purpose",
        brain,
        tools=(_read_tool(),),
        context_policy=policy,
        llm_provider=REAL_PROVIDER,
    )
    return list(run(agent, state, max_turns=n_reads + 2))


def _tool_compact() -> ContextPolicy:
    return ContextPolicy(strategy=ToolCompactStrategy(threshold_tokens=THRESHOLD))


class BoundedContextTest(unittest.TestCase):
    def test_baseline_grows_unbounded_but_compression_stays_bounded(self) -> None:
        # The core claim: doubling the work roughly doubles the active context
        # without compression, but barely moves it with compression.
        base_small = summarize_compression(_run_tool_heavy(ContextPolicy(), 6))
        base_large = summarize_compression(_run_tool_heavy(ContextPolicy(), 12))
        comp_small = summarize_compression(_run_tool_heavy(_tool_compact(), 6))
        comp_large = summarize_compression(_run_tool_heavy(_tool_compact(), 12))

        # Baseline: never compacts, peak == final, and grows ~linearly.
        self.assertEqual(base_large.compactions, 0)
        self.assertEqual(base_small.peak_active_tokens, base_small.final_active_tokens)
        self.assertGreater(
            base_large.peak_active_tokens / base_small.peak_active_tokens, 1.7
        )

        # Compression: peak is flat across the two sizes (bounded).
        self.assertGreater(comp_large.compactions, comp_small.compactions)
        self.assertLess(
            comp_large.peak_active_tokens / comp_small.peak_active_tokens, 1.2
        )
        # And the bounded peak is far below the unbounded one at scale.
        self.assertLess(
            comp_large.peak_active_tokens, 0.5 * base_large.peak_active_tokens
        )

    def test_compression_keeps_peak_near_the_threshold(self) -> None:
        metrics = summarize_compression(_run_tool_heavy(_tool_compact(), 12))
        # Peak is the trigger threshold plus at most one over-budget turn's
        # growth, not an unbounded pile — concretely well under 2x threshold.
        self.assertLess(metrics.peak_active_tokens, 2 * THRESHOLD)


class DensityTest(unittest.TestCase):
    def test_every_fold_reduces_the_active_context(self) -> None:
        events = _run_tool_heavy(_tool_compact(), 12)
        folds = [e for e in events if isinstance(e, ContextCompressionEvent)]
        self.assertGreater(len(folds), 0)
        for fold in folds:
            self.assertLess(
                fold.after_tokens,
                fold.before_tokens,
                msg=f"fold did not shrink: {fold}",
            )
        # Aggregate density: each fold keeps well under the full active context.
        metrics = summarize_compression(events)
        self.assertLess(metrics.mean_kept_fraction, 0.8)
        self.assertGreater(metrics.tokens_dropped, 0)

    def test_task_message_is_never_folded(self) -> None:
        # Index 0 is the task (a preserved kind) — durable across every fold.
        events = _run_tool_heavy(_tool_compact(), 12)
        for fold in (e for e in events if isinstance(e, ContextCompressionEvent)):
            self.assertNotIn(0, fold.compressed_message_indices)


class RecoverabilityTest(unittest.TestCase):
    def test_a_folded_fact_is_recoverable_via_recall(self) -> None:
        # A summarize fold drops an early message; the cited originals are still
        # retrievable from the append-only transcript by index.
        secret = "ACCESS-CODE-9931"

        def brain(visible: list[Message]) -> Message:
            if any(m.kind == "summary" for m in visible):
                return assistant_message(
                    "done", sender="w", target="user", kind="final"
                )
            return assistant_message("noted", sender="w", target="user", kind="step")

        def compressor(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "Summary: earlier notes condensed (code omitted).",
                sender="c",
                target="runtime",
                kind="final",
            )

        state = State("task")
        state.send("task", "user", "w", "task")  # 0
        state.send("message", "user", "w", f"the code is {secret}")  # 1
        state.send("message", "user", "w", "filler " + "x" * 600)  # 2
        policy = ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=Agent("c", compressor),
                threshold_tokens=50,
                keep_recent=1,
            )
        )
        events = list(run(Agent("w", brain, context_policy=policy), state, max_turns=3))

        fold = next(e for e in events if isinstance(e, ContextCompressionEvent))
        self.assertIn(1, fold.compressed_message_indices)
        # Folded out of the live view...
        active = [text_of(m.content) for m in state.active_context_messages()]
        self.assertFalse(any(secret in text for text in active))
        # ...but recall returns it from the cited indices.
        recall = make_recall_tool(state)
        result = recall.execute(
            "call", {"indices": fold.compressed_message_indices}, _no_abort, None
        )
        self.assertFalse(result.is_error)
        self.assertIn(secret, tool_result_text(result))


class ToolPairSafetyTest(unittest.TestCase):
    def test_summarize_compressor_input_does_not_split_tool_pairs(self) -> None:
        captured: dict[str, list[Message]] = {}

        def compressor(visible: list[Message]) -> Message:
            captured["visible"] = visible
            return assistant_message(
                "summary", sender="c", target="runtime", kind="final"
            )

        active = [
            (0, assistant_message("task", sender="user", target="w", kind="task")),
            (
                1,
                assistant_message(
                    [ToolCallBlock("call_split", "read_file", {"path": "a.py"})],
                    sender="w",
                    target="user",
                    kind="step",
                ),
            ),
            (
                2,
                tool_results_message(
                    [
                        ToolResultBlock(
                            "call_split",
                            "read_file",
                            (TextBlock("file contents"),),
                        )
                    ],
                    target="w",
                ),
            ),
        ]
        strategy = SummarizeStrategy(
            compressor=Agent("c", compressor),
            threshold_tokens=1,
            keep_recent=1,
        )

        decision = strategy(active, "w")

        self.assertIsNone(decision)
        self.assertNotIn("visible", captured)
        seen = captured.get("visible", [])
        seen_calls = {
            call.id for message in seen for call in message_tool_calls(message)
        }
        seen_results = {
            block.tool_call_id
            for message in seen
            for block in tool_results_of(message.content)
        }
        self.assertEqual(seen_calls, seen_results)

    def test_summarize_compressor_input_drops_split_boundary_pair_only(self) -> None:
        captured: dict[str, list[Message]] = {}

        def compressor(visible: list[Message]) -> Message:
            captured["visible"] = visible
            return assistant_message(
                "summary", sender="c", target="runtime", kind="final"
            )

        active = [
            (0, assistant_message("task", sender="user", target="w", kind="task")),
            (
                1,
                assistant_message(
                    [ToolCallBlock("call_old", "read_file", {"path": "old.py"})],
                    sender="w",
                    target="user",
                    kind="step",
                ),
            ),
            (
                2,
                tool_results_message(
                    [
                        ToolResultBlock(
                            "call_old",
                            "read_file",
                            (TextBlock("old contents"),),
                        )
                    ],
                    target="w",
                ),
            ),
            (
                3,
                assistant_message(
                    [ToolCallBlock("call_new", "read_file", {"path": "new.py"})],
                    sender="w",
                    target="user",
                    kind="step",
                ),
            ),
            (
                4,
                tool_results_message(
                    [
                        ToolResultBlock(
                            "call_new",
                            "read_file",
                            (TextBlock("new contents"),),
                        )
                    ],
                    target="w",
                ),
            ),
            (5, assistant_message("tail", sender="w", target="user", kind="step")),
        ]
        strategy = SummarizeStrategy(
            compressor=Agent("c", compressor),
            threshold_tokens=1,
            keep_recent=2,
        )

        decision = strategy(active, "w")

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.compress_indices, (1, 2))
        seen = captured["visible"]
        seen_calls = {
            call.id for message in seen for call in message_tool_calls(message)
        }
        seen_results = {
            block.tool_call_id
            for message in seen
            for block in tool_results_of(message.content)
        }
        self.assertEqual(seen_calls, {"call_old"})
        self.assertEqual(seen_results, {"call_old"})

    def test_summarize_compresses_one_contiguous_span_around_preserved_task(
        self,
    ) -> None:
        captured: dict[str, list[Message]] = {}

        def compressor(visible: list[Message]) -> Message:
            captured["visible"] = visible
            return assistant_message(
                "summary", sender="c", target="runtime", kind="final"
            )

        active = [
            (0, user_message("prior task now message " + "x" * 1200, target="w")),
            (1, assistant_message("prior work " + "y" * 1200, target="user")),
            (2, user_message("current task", target="w", kind="task")),
            (3, assistant_message("current early work " + "z" * 100, target="user")),
            (4, assistant_message("recent one", target="user")),
            (5, assistant_message("recent two", target="user")),
        ]
        strategy = SummarizeStrategy(
            compressor=Agent("c", compressor),
            threshold_tokens=1,
            keep_recent=2,
            preserve_kinds=("task", "system", "context"),
        )

        decision = strategy(active, "w")

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.compress_indices, (0, 1))
        self.assertEqual(
            [text_of(message.content) for message in captured["visible"]],
            [
                "prior task now message " + "x" * 1200,
                "prior work " + "y" * 1200,
                (
                    "Compact the older conversation above into working memory "
                    "for agent 'w'. It replaces those messages; the task and "
                    "recent messages stay visible.\n\n"
                    "Write a terse, self-contained summary under these "
                    "headings, omitting any that have nothing:\n"
                    "- Goal: the objective plus constraints / acceptance "
                    "criteria.\n"
                    "- Done: what has been accomplished and verified.\n"
                    "- State: where the work stands now.\n"
                    "- Facts & identifiers: key facts, decisions, and tool "
                    "results, with exact paths, symbols, commands, errors, "
                    "test names, and values kept VERBATIM.\n"
                    "- Open: unresolved questions and blockers.\n"
                    "- Next: the next concrete action(s).\n"
                    "- Tried & rejected: approaches already attempted that "
                    "failed, and why.\n\n"
                    "Preserve facts exactly; never invent. Omit low-value "
                    "wording."
                ),
            ],
        )

    def test_summarize_runtime_applies_two_spans_around_current_task(
        self,
    ) -> None:
        captured: list[list[str]] = []

        def compressor(visible: list[Message]) -> Message:
            captured.append([text_of(message.content) for message in visible])
            return assistant_message(
                f"summary {len(captured)}",
                sender="c",
                target="runtime",
                kind="final",
            )

        state = State("chain")
        state.record(user_message("prior task now message " + "x" * 1200, target="w"))
        state.record(
            assistant_message("prior work " + "y" * 1200, sender="w", target="user")
        )
        state.record(user_message("current task", target="w", kind="task"))
        state.record(
            assistant_message(
                "current early work " + "z" * 1200,
                sender="w",
                target="user",
            )
        )
        state.record(assistant_message("recent one", sender="w", target="user"))
        state.record(assistant_message("recent two", sender="w", target="user"))
        policy = ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=Agent("c", compressor),
                threshold_tokens=1,
                keep_recent=2,
                preserve_kinds=("task", "system", "context"),
            )
        )

        events = maybe_compress_context(
            Agent(
                "w", lambda visible: assistant_message("done"), context_policy=policy
            ),
            state,
            policy,
        )

        folds = [
            event for event in events if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(len(folds), 2)
        self.assertEqual(
            [fold.compressed_message_indices for fold in folds],
            [[0, 1], [3]],
        )
        self.assertEqual(
            [message.kind for message in state.active_context_messages()],
            ["summary", "task", "summary", "message", "message"],
        )
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0][0], "prior task now message " + "x" * 1200)
        self.assertEqual(captured[0][1], "prior work " + "y" * 1200)
        self.assertEqual(captured[1][0], "current early work " + "z" * 1200)
        self.assertNotIn("current task", captured[0])
        self.assertNotIn("current task", captured[1])
        elapsed = [event.elapsed for event in state.events]
        self.assertEqual(elapsed, sorted(elapsed))
        self.assertTrue(all(fold.start_elapsed <= fold.elapsed for fold in folds))

    def test_summarize_runtime_skips_empty_current_early_span(self) -> None:
        captured: list[list[str]] = []

        def compressor(visible: list[Message]) -> Message:
            captured.append([text_of(message.content) for message in visible])
            return assistant_message(
                "summary", sender="c", target="runtime", kind="final"
            )

        state = State("chain")
        state.record(user_message("prior task now message " + "x" * 1200, target="w"))
        state.record(
            assistant_message("prior work " + "y" * 1200, sender="w", target="user")
        )
        state.record(user_message("current task", target="w", kind="task"))
        state.record(assistant_message("recent one", sender="w", target="user"))
        state.record(assistant_message("recent two", sender="w", target="user"))
        policy = ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=Agent("c", compressor),
                threshold_tokens=1,
                keep_recent=2,
                preserve_kinds=("task", "system", "context"),
            )
        )

        events = maybe_compress_context(
            Agent(
                "w", lambda visible: assistant_message("done"), context_policy=policy
            ),
            state,
            policy,
        )

        folds = [
            event for event in events if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].compressed_message_indices, [0, 1])
        self.assertEqual(
            [message.kind for message in state.active_context_messages()],
            ["summary", "task", "message", "message"],
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], "prior task now message " + "x" * 1200)
        self.assertEqual(captured[0][1], "prior work " + "y" * 1200)
        self.assertNotIn("current task", captured[0])


def _run_text_heavy(policy: ContextPolicy, n_steps: int) -> list:
    """Emit `n_steps` big text messages (no tool calls) under `policy`."""
    turn = {"n": 0}

    def brain(visible: list[Message]) -> Message:
        del visible
        index = turn["n"]
        turn["n"] += 1
        if index < n_steps:
            # Big enough that a few steps cross THRESHOLD with no tool exchanges,
            # so the tiered policy must fall through to the summarize stage.
            return assistant_message(
                "analysis paragraph " + "elaboration " * 400,
                sender="analyst",
                target="user",
                kind="step",
            )
        return assistant_message("done", sender="analyst", target="user", kind="final")

    state = State("analyze")
    state.send("task", "user", "analyst", state.task)
    agent = Agent("analyst", brain, context_policy=policy, llm_provider=REAL_PROVIDER)
    return list(run(agent, state, max_turns=n_steps + 2))


def _summarizer() -> Agent:
    def compressor(visible: list[Message]) -> Message:
        del visible
        return assistant_message(
            "condensed", sender="c", target="runtime", kind="final"
        )

    return Agent("c", compressor)


class StrategyAttributionTest(unittest.TestCase):
    """Each fold event names the strategy that produced it, so a fold is
    attributable without sniffing the summary text — the gap this closes."""

    def test_rule_based_folds_are_labeled_tool_compact(self) -> None:
        events = _run_tool_heavy(_tool_compact(), 8)
        folds = [e for e in events if isinstance(e, ContextCompressionEvent)]
        self.assertTrue(folds)
        self.assertTrue(all(f.strategy == "tool-compact" for f in folds))
        metrics = summarize_compression(events)
        self.assertEqual(metrics.folds_by_strategy, {"tool-compact": len(folds)})

    def test_summarize_folds_are_labeled_summarize(self) -> None:
        policy = ContextPolicy(
            strategy=SummarizeStrategy(
                compressor=_summarizer(), threshold_tokens=THRESHOLD, keep_recent=2
            )
        )
        folds = [
            e
            for e in _run_text_heavy(policy, 8)
            if isinstance(e, ContextCompressionEvent)
        ]
        self.assertTrue(folds)
        self.assertTrue(all(f.strategy == "summarize" for f in folds))

    def test_tiered_event_names_the_stage_that_actually_fired(self) -> None:
        # One TieredStrategy policy, two scenarios: the event self-identifies
        # which stage fired — the cheap rule-based fold on tool-heavy work, the
        # LLM summary fallback on text-only work. Previously indistinguishable.
        def tiered() -> ContextPolicy:
            return ContextPolicy(
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
            )

        tool_metrics = summarize_compression(_run_tool_heavy(tiered(), 8))
        text_metrics = summarize_compression(_run_text_heavy(tiered(), 8))
        self.assertEqual(set(tool_metrics.folds_by_strategy), {"tool-compact"})
        self.assertEqual(set(text_metrics.folds_by_strategy), {"summarize"})


class MetricsShapeTest(unittest.TestCase):
    def test_no_compression_run_reports_zero_folds_and_full_keep(self) -> None:
        metrics = summarize_compression(_run_tool_heavy(ContextPolicy(), 4))
        self.assertEqual(metrics.compactions, 0)
        self.assertEqual(metrics.tokens_dropped, 0)
        self.assertEqual(metrics.mean_kept_fraction, 1.0)
        self.assertGreater(metrics.transcript_messages, 0)
        self.assertEqual(
            metrics.peak_active_tokens, metrics.final_active_tokens
        )  # monotonic, never shrinks


if __name__ == "__main__":
    unittest.main()
