"""LLM proposer, EVOLVE-BLOCK utilities, and agent evaluator tests."""

from __future__ import annotations

import random
import unittest

from simple_agent_lab.core import Agent
from simple_agent_lab.evolve import (
    Candidate,
    Evaluation,
    EvolutionRecord,
    agent_task_evaluator,
    build_mutation_prompt,
    check_immutable_regions,
    evolve_blocks,
    llm_propose,
    parse_fields,
    render_fields,
    replace_evolve_blocks,
)
from simple_agent_lab.messages import AssistantMessage, TextBlock


def make_record(payload, fitness=0.5, feedback="") -> EvolutionRecord:
    return EvolutionRecord(
        candidate=Candidate(id="c0000", payload=payload),
        evaluation=Evaluation(fitness=fitness, feedback=feedback),
        accepted=True,
        reason="test",
    )


class FieldWireFormatTest(unittest.TestCase):
    def test_render_then_parse_round_trips(self):
        payload = {"system_prompt": "line one\nline two", "code": "def f():\n    pass"}
        rendered = render_fields(payload, ["system_prompt", "code"])
        parsed = parse_fields(rendered, ["system_prompt", "code"])
        self.assertEqual(parsed, payload)

    def test_parse_tolerates_language_tag_and_prose(self):
        response = (
            "I made the prompt stricter.\n\n"
            "### system_prompt\n```text\nBe terse.\n```\nDone.\n"
        )
        self.assertEqual(
            parse_fields(response, ["system_prompt"]),
            {"system_prompt": "Be terse."},
        )

    def test_parse_raises_when_no_field_found(self):
        with self.assertRaises(ValueError):
            parse_fields("no blocks here", ["system_prompt"])


class LLMProposeTest(unittest.TestCase):
    def test_prompt_carries_parent_feedback_and_inspirations(self):
        parent = make_record({"system_prompt": "old"}, 0.4, feedback="failed task 2")
        inspiration = make_record({"system_prompt": "shiny"}, 0.9)
        prompt = build_mutation_prompt(
            [parent, inspiration], task="win", fields=["system_prompt"]
        )
        for expected in ("win", "old", "failed task 2", "shiny", "0.9000"):
            self.assertIn(expected, prompt)

    def test_changed_fields_override_and_others_inherit(self):
        def ask(prompt: str) -> str:
            return "Tightened.\n\n### system_prompt\n```\nnew\n```"

        propose = llm_propose(ask, task="win", fields=["system_prompt"])
        parent = make_record({"system_prompt": "old", "temperature": 0.3})
        proposal = propose([parent], random.Random(0))
        self.assertEqual(proposal.payload, {"system_prompt": "new", "temperature": 0.3})
        self.assertEqual(proposal.operator, "llm_mutate")
        self.assertEqual(proposal.note, "Tightened.")

    def test_unusable_response_raises(self):
        propose = llm_propose(lambda p: "sorry, no", task="win", fields=["x"])
        with self.assertRaises(ValueError):
            propose([make_record({"x": "1"})], random.Random(0))


SOURCE = """import math
# EVOLVE-BLOCK-START
def h(x):
    return x
# EVOLVE-BLOCK-END
def main():
    return h(2)
"""


class EvolveBlockTest(unittest.TestCase):
    def test_extract_and_replace_round_trip(self):
        blocks = evolve_blocks(SOURCE)
        self.assertEqual(blocks, ["def h(x):\n    return x\n"])
        self.assertEqual(replace_evolve_blocks(SOURCE, blocks), SOURCE)

    def test_replace_touches_only_the_block(self):
        mutated = replace_evolve_blocks(SOURCE, ["def h(x):\n    return x * x\n"])
        self.assertIn("return x * x", mutated)
        self.assertIn("def main():", mutated)
        check_immutable_regions(SOURCE, mutated)  # must not raise

    def test_multiple_blocks(self):
        source = (
            "# EVOLVE-BLOCK-START\na\n# EVOLVE-BLOCK-END\n"
            "fixed\n"
            "# EVOLVE-BLOCK-START\nb\n# EVOLVE-BLOCK-END\n"
        )
        self.assertEqual(evolve_blocks(source), ["a\n", "b\n"])
        replaced = replace_evolve_blocks(source, ["A\n", "B\n"])
        self.assertEqual(evolve_blocks(replaced), ["A\n", "B\n"])
        check_immutable_regions(source, replaced)

    def test_scaffold_edit_detected(self):
        mutated = replace_evolve_blocks(SOURCE, ["def h(x):\n    return 9\n"])
        with self.assertRaises(ValueError):
            check_immutable_regions(SOURCE, mutated.replace("main", "main2"))

    def test_unbalanced_markers_raise(self):
        for bad in (
            "# EVOLVE-BLOCK-START\nno end",
            "# EVOLVE-BLOCK-END\nno start",
            "# EVOLVE-BLOCK-START\na\n# EVOLVE-BLOCK-END\n# EVOLVE-BLOCK-END\n",
        ):
            with self.assertRaises(ValueError):
                evolve_blocks(bad)

    def test_replacement_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            replace_evolve_blocks(SOURCE, ["a", "b"])


class AgentTaskEvaluatorTest(unittest.TestCase):
    def _echo_agent_factory(self):
        """Agent answers correctly only when the prompt payload says 'terse'."""

        def build_agent(candidate: Candidate) -> Agent:
            terse = "terse" in candidate.payload["system_prompt"]

            def generate(visible) -> AssistantMessage:
                text = "4" if terse else "the answer is 4"
                return AssistantMessage(
                    content=(TextBlock(text=text),),
                    sender="agent",
                    target="user",
                    kind="final",
                )

            return Agent(name="agent", generate=generate)

        return build_agent

    def test_fitness_is_mean_and_feedback_quotes_failures(self):
        evaluate = agent_task_evaluator(
            self._echo_agent_factory(),
            ["2+2?", "8/2?"],
            lambda task, output: 1.0 if output.strip() == "4" else 0.0,
            max_turns=2,
        )
        verbose = evaluate(Candidate(id="c0000", payload={"system_prompt": "chatty"}))
        self.assertEqual(verbose.fitness, 0.0)
        self.assertIn("the answer is 4", verbose.feedback)
        self.assertEqual(verbose.metrics["solved"], 0)

        terse = evaluate(Candidate(id="c0001", payload={"system_prompt": "be terse"}))
        self.assertEqual(terse.fitness, 1.0)
        self.assertEqual(terse.feedback, "")
        self.assertEqual(terse.metrics["solved"], 2)

    def test_empty_task_list_raises(self):
        with self.assertRaises(ValueError):
            agent_task_evaluator(self._echo_agent_factory(), [], lambda t, o: 0.0)


if __name__ == "__main__":
    unittest.main()
