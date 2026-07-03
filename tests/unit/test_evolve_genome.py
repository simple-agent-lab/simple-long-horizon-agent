"""Component-layer tests: genome schema, operators, island selection."""

from __future__ import annotations

import json
import random
import unittest

from simple_agent_lab.evolve import (
    Archive,
    Candidate,
    ComponentSpec,
    Evaluation,
    EvolutionRecord,
    GenomeSpec,
    Proposal,
    agent_genome,
    build_genome_agent,
    crossover_propose,
    genome_propose,
    mix_operators,
    seed_agent_payload,
    select_islands,
    validate_component,
)
from simple_agent_lab.llm import Provider
from simple_agent_lab.tools import AgentTool, text_result

FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def make_record(payload, fitness=0.5, *, candidate_id="c0000", parent_ids=()):
    return EvolutionRecord(
        candidate=Candidate(id=candidate_id, payload=payload, parent_ids=parent_ids),
        evaluation=Evaluation(fitness=fitness),
        accepted=True,
        reason="test",
    )


class GenomeSpecTest(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            GenomeSpec(
                components=(
                    ComponentSpec(key="a", description="x"),
                    ComponentSpec(key="a", description="y"),
                )
            )

    def test_all_immutable_rejected(self):
        with self.assertRaises(ValueError):
            GenomeSpec(
                components=(ComponentSpec(key="a", description="x", mutable=False),)
            )

    def test_unknown_component_lookup_raises(self):
        spec = GenomeSpec(components=(ComponentSpec(key="a", description="x"),))
        with self.assertRaises(KeyError):
            spec.component("b")


class ValidateComponentTest(unittest.TestCase):
    def test_json_kind_must_parse(self):
        spec = ComponentSpec(key="cfg", description="", kind="json")
        validate_component(spec, '{"k": 1}')
        with self.assertRaises(ValueError) as ctx:
            validate_component(spec, "not json")
        self.assertIn("cfg", str(ctx.exception))

    def test_code_kind_guards_evolve_block_scaffold(self):
        parent = (
            "# EVOLVE-BLOCK-START\nreturn 1\n# EVOLVE-BLOCK-END\ndef scaffold(): ...\n"
        )
        spec = ComponentSpec(key="code", description="", kind="code")
        mutated_ok = parent.replace("return 1", "return 2")
        validate_component(spec, mutated_ok, parent_value=parent)
        with self.assertRaises(ValueError):
            validate_component(
                spec, mutated_ok.replace("scaffold", "hacked"), parent_value=parent
            )

    def test_custom_validate_runs_last(self):
        def no_shouting(value: str) -> None:
            if value.isupper():
                raise ValueError("no shouting")

        spec = ComponentSpec(key="p", description="", validate=no_shouting)
        validate_component(spec, "quiet")
        with self.assertRaises(ValueError):
            validate_component(spec, "LOUD")


class GenomeProposeTest(unittest.TestCase):
    def _spec(self):
        return GenomeSpec(
            components=(
                ComponentSpec(key="system_prompt", description="the prompt"),
                ComponentSpec(key="config", description="settings", kind="json"),
                ComponentSpec(key="frozen", description="fixed", mutable=False),
            ),
            task="win the benchmark",
        )

    def test_targets_only_mutable_components_and_docs_in_prompt(self):
        prompts: list[str] = []

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return "note\n\n### system_prompt\n```\nnew\n```"

        propose = genome_propose(ask, self._spec(), components_per_mutation=2)
        parent = make_record({"system_prompt": "old", "config": "{}", "frozen": "keep"})
        proposal = propose([parent], random.Random(0))
        self.assertEqual(proposal.payload["frozen"], "keep")
        self.assertEqual(proposal.payload["system_prompt"], "new")
        self.assertIn("win the benchmark", prompts[0])
        self.assertIn("the prompt", prompts[0])  # component docs included
        self.assertNotIn("- frozen", prompts[0])  # immutable never offered

    def test_invalid_mutation_raises(self):
        def ask(prompt: str) -> str:
            return "### config\n```\nnot json\n```"

        propose = genome_propose(
            ask,
            GenomeSpec(
                components=(ComponentSpec(key="config", description="", kind="json"),),
                task="t",
            ),
        )
        parent = make_record({"config": "{}"})
        with self.assertRaises(ValueError):
            propose([parent], random.Random(0))


class AgentGenomeTest(unittest.TestCase):
    def _tool(self, name="lookup", description="original description"):
        return AgentTool(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}},
            execute=lambda call_id, args, abort, on_update=None: text_result("ok"),
        )

    def test_agent_genome_components_depend_on_tools(self):
        bare = agent_genome(task="t")
        self.assertEqual(bare.mutable_keys(), ("system_prompt", "instructions"))
        with_tools = agent_genome(task="t", tools=[self._tool()])
        self.assertIn("tool_descriptions", with_tools.mutable_keys())
        self.assertIn("lookup", with_tools.component("tool_descriptions").description)

    def test_build_genome_agent_applies_prompt_and_tool_overrides(self):
        tool = self._tool()
        payload = seed_agent_payload("Base prompt.", tools=[tool])
        payload["instructions"] = "Lesson: check twice."
        payload["tool_descriptions"] = json.dumps({"lookup": "better description"})
        agent = build_genome_agent(
            Candidate(id="c0000", payload=payload),
            provider=FAKE_PROVIDER,
            tools=[tool],
        )
        self.assertIn("Base prompt.", agent.system_prompt)
        self.assertIn("Lesson: check twice.", agent.system_prompt)
        self.assertEqual(agent.tools[0].description, "better description")
        # The original tool value is untouched (replace, not mutate).
        self.assertEqual(tool.description, "original description")

    def test_unknown_tool_override_raises(self):
        payload = seed_agent_payload("p", tools=[self._tool()])
        payload["tool_descriptions"] = json.dumps({"ghost": "boo"})
        with self.assertRaises(ValueError) as ctx:
            build_genome_agent(
                Candidate(id="c0000", payload=payload),
                provider=FAKE_PROVIDER,
                tools=[self._tool()],
            )
        self.assertIn("ghost", str(ctx.exception))


class OperatorsTest(unittest.TestCase):
    def test_mix_operators_samples_by_weight_and_keeps_labels(self):
        def op(name):
            def propose(parents, rng):
                return Proposal(payload={"x": name}, operator=name)

            return propose

        mixed = mix_operators([(op("often"), 0.9), (op("rare"), 0.1)])
        rng = random.Random(0)
        labels = [mixed([make_record({"x": "seed"})], rng).operator for _ in range(200)]
        self.assertGreater(labels.count("often"), labels.count("rare"))
        self.assertEqual(set(labels), {"often", "rare"})

    def test_mix_operators_validates_inputs(self):
        with self.assertRaises(ValueError):
            mix_operators([])
        with self.assertRaises(ValueError):
            mix_operators([(lambda p, r: None, -1.0)])

    def test_crossover_combines_parent_and_inspiration(self):
        prompts: list[str] = []

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return "took A's tone, B's format\n\n### p\n```\nmerged\n```"

        propose = crossover_propose(ask, task="t", fields=["p"])
        parent = make_record({"p": "alpha"}, 0.4, candidate_id="c0001")
        inspiration = make_record({"p": "beta"}, 0.8, candidate_id="c0002")
        proposal = propose([parent, inspiration], random.Random(0))
        self.assertEqual(proposal.payload, {"p": "merged"})
        self.assertEqual(proposal.operator, "llm_crossover")
        self.assertIn("alpha", prompts[0])
        self.assertIn("beta", prompts[0])

    def test_crossover_single_parent_falls_back_to_mutation(self):
        propose = crossover_propose(
            lambda p: "### p\n```\nsolo child\n```", task="t", fields=["p"]
        )
        proposal = propose([make_record({"p": "alpha"})], random.Random(0))
        self.assertEqual(proposal.payload, {"p": "solo child"})
        self.assertEqual(proposal.operator, "llm_crossover_solo")


class SelectIslandsTest(unittest.TestCase):
    def _archive_with_islands(self):
        """Two seeds -> two islands; one child on each island."""

        archive = Archive()
        archive.add(make_record({"x": "s0"}, 0.5, candidate_id="c0000"))
        archive.add(make_record({"x": "s1"}, 0.6, candidate_id="c0001"))
        archive.add(
            make_record({"x": "k0"}, 0.7, candidate_id="c0002", parent_ids=("c0000",))
        )
        archive.add(
            make_record({"x": "k1"}, 0.9, candidate_id="c0003", parent_ids=("c0001",))
        )
        return archive

    def test_parent_and_inspirations_stay_on_one_island(self):
        archive = self._archive_with_islands()
        # len(records)=4, num_islands=2 -> island 0 = {c0000, c0002}.
        select = select_islands(num_islands=2, inspirations=2, migration_interval=0)
        parents = select(archive, random.Random(0))
        island_zero = {"c0000", "c0002"}
        for record in parents:
            self.assertIn(record.candidate.id, island_zero)

    def test_migration_round_offers_global_inspirations(self):
        archive = self._archive_with_islands()
        # migration_interval=5: migrate when len(records) % 5 == 4 -> now.
        select = select_islands(num_islands=2, inspirations=1, migration_interval=5)
        parents = select(archive, random.Random(0))
        # Best global record (c0003, island 1) crosses over as inspiration.
        self.assertIn("c0003", [r.candidate.id for r in parents[1:]])

    def test_empty_island_borrows_whole_population(self):
        archive = Archive()
        archive.add(make_record({"x": "s0"}, 0.5, candidate_id="c0000"))
        # len(records)=1, island 1 selected but only island 0 is populated.
        select = select_islands(num_islands=2, inspirations=0)
        parents = select(archive, random.Random(0))
        self.assertEqual(parents[0].candidate.id, "c0000")

    def test_invalid_island_count_raises(self):
        with self.assertRaises(ValueError):
            select_islands(num_islands=0)


if __name__ == "__main__":
    unittest.main()
