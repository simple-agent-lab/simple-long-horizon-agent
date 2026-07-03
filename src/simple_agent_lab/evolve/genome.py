"""Typed genome components: declare WHAT evolves, the harness does the rest.

The bare loop treats a payload as an opaque mapping — maximally flexible,
but every experiment then re-solves the same three problems by hand: how a
component is described to the mutation model, which mutations are illegal,
and how a payload becomes a runnable `Agent`. This module is that reusable
layer (the "materialize the harness as editable components" idea from
harness-evolution work, arXiv:2604.25850):

- `ComponentSpec` / `GenomeSpec` — a typed schema over the payload: each
  evolvable component has a kind (text / code / json), a description shown
  to the proposer, a mutability flag, and validation.
- `genome_propose` — a schema-aware `ProposeFn`: samples which mutable
  component(s) to target, includes the component docs in the prompt, and
  validates every returned value (JSON must parse; EVOLVE-BLOCK code must
  keep its scaffold; custom rules run last). Illegal mutations raise, so the
  loop records them as rejected candidates instead of scoring them.
- `agent_genome` / `build_genome_agent` — a standard genome for the common
  experiment "evolve the agent definition itself": system prompt, an
  appended instructions block (lessons/procedures), and per-tool description
  overrides, built into a live `Agent` on the existing `make_llm_agent`.

Everything here is optional sugar over the same four seams; a custom
experiment can keep passing plain payloads and its own callables.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AgentTool

from .code_blocks import EVOLVE_START, check_immutable_regions, evolve_blocks
from .propose import AskFn, proposal_note, build_mutation_prompt, parse_fields
from .types import Candidate, EvolutionRecord, Payload, Proposal, ProposeFn

ComponentKind = Literal["text", "code", "json"]

# Extra per-component rule; raise ValueError on illegal content.
ValidateFn = Callable[[str], None]


@dataclass(frozen=True)
class ComponentSpec:
    """One evolvable slot in the genome.

    `description` is not documentation for humans — it is shown verbatim to
    the proposer model, so it is where an experiment steers what kinds of
    edits make sense for this component. `mutable=False` components ride
    along in the payload (visible, inherited) but are never offered for
    mutation. `validate` runs after the kind check.
    """

    key: str
    description: str
    kind: ComponentKind = "text"
    mutable: bool = True
    validate: ValidateFn | None = None


@dataclass(frozen=True)
class GenomeSpec:
    """The typed schema for a payload: which components exist, plus the goal.

    `task` is the optimization objective sentence used by `genome_propose`
    prompts, kept on the spec so the schema and the goal travel together.
    """

    components: tuple[ComponentSpec, ...]
    task: str = ""

    def __post_init__(self) -> None:
        keys = [component.key for component in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate component keys: {keys}")
        if not any(component.mutable for component in self.components):
            raise ValueError("a genome needs at least one mutable component")

    def component(self, key: str) -> ComponentSpec:
        for component in self.components:
            if component.key == key:
                return component
        raise KeyError(
            f"no component {key!r}; known: {[c.key for c in self.components]}"
        )

    def mutable_keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.components if c.mutable)


def validate_component(
    spec: ComponentSpec, value: str, *, parent_value: str = ""
) -> None:
    """Kind-specific legality check, then the component's custom rule.

    - `json`: must parse as JSON.
    - `code`: when the parent used EVOLVE-BLOCK markers, the mutation must
      keep the scaffold outside the blocks byte-identical
      (`check_immutable_regions`); marker-free code is only checked for
      balanced markers.
    - `text`: no kind rule.

    Raises `ValueError` naming the component so a rejected record's error is
    actionable.
    """

    try:
        if spec.kind == "json":
            json.loads(value)
        elif spec.kind == "code":
            if parent_value and EVOLVE_START in parent_value:
                check_immutable_regions(parent_value, value)
            else:
                evolve_blocks(value)  # balanced-marker check only
        if spec.validate is not None:
            spec.validate(value)
    except ValueError as exc:
        raise ValueError(f"component {spec.key!r}: {exc}") from exc


def validate_payload(
    spec: GenomeSpec, payload: Payload, *, parent_payload: Payload | None = None
) -> None:
    """Validate every declared component present in `payload`."""

    parent = dict(parent_payload or {})
    for component in spec.components:
        if component.key in payload:
            validate_component(
                component,
                str(payload[component.key]),
                parent_value=str(parent.get(component.key, "")),
            )


def _component_docs(spec: GenomeSpec, keys: Sequence[str]) -> str:
    lines = ["The candidate's components you may edit this round:"]
    for key in keys:
        component = spec.component(key)
        lines.append(f"- {component.key} ({component.kind}): {component.description}")
    return "\n".join(lines)


def genome_propose(
    ask: AskFn,
    spec: GenomeSpec,
    *,
    components_per_mutation: int = 1,
    guidance: str = "",
    operator: str = "genome_mutate",
) -> ProposeFn:
    """A schema-aware `ProposeFn` over a `GenomeSpec`.

    Each call samples `components_per_mutation` mutable components to target
    (focused edits attribute fitness changes to specific components), shows
    their docs alongside the shared mutation prompt, and validates whatever
    comes back before it becomes a candidate. Untargeted and immutable
    components inherit from the parent.
    """

    def propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
        mutable = list(spec.mutable_keys())
        chosen = sorted(
            rng.sample(mutable, min(components_per_mutation, len(mutable))),
            key=mutable.index,
        )
        docs = _component_docs(spec, chosen)
        prompt = build_mutation_prompt(
            parents,
            task=spec.task,
            fields=chosen,
            guidance=f"{guidance}\n\n{docs}" if guidance else docs,
        )
        response = ask(prompt)
        changed = parse_fields(response, chosen)
        parent_payload = dict(parents[0].candidate.payload)
        for key, value in changed.items():
            validate_component(
                spec.component(key),
                value,
                parent_value=str(parent_payload.get(key, "")),
            )
        payload = {**parent_payload, **changed}
        return Proposal(
            payload=payload, operator=operator, note=proposal_note(response)
        )

    return propose


# Standard agent-genome component keys (the common "evolve the agent
# definition" experiment). `build_genome_agent` understands exactly these.
SYSTEM_PROMPT_KEY = "system_prompt"
INSTRUCTIONS_KEY = "instructions"
TOOL_DESCRIPTIONS_KEY = "tool_descriptions"


def agent_genome(*, task: str, tools: Sequence[AgentTool] = ()) -> GenomeSpec:
    """The standard genome for evolving an agent definition.

    Components: the system prompt, an `instructions` block appended after it
    (the natural home for evolved lessons/procedures), and — when the agent
    has tools — a JSON mapping of per-tool description overrides (evolving
    how tools are explained is one of the highest-leverage harness edits).
    """

    components = [
        ComponentSpec(
            key=SYSTEM_PROMPT_KEY,
            description="The agent's system prompt: role, behavior, output format.",
        ),
        ComponentSpec(
            key=INSTRUCTIONS_KEY,
            description=(
                "Extra instructions appended after the system prompt: lessons "
                "learned, procedures, edge-case handling."
            ),
        ),
    ]
    if tools:
        names = ", ".join(tool.name for tool in tools)
        components.append(
            ComponentSpec(
                key=TOOL_DESCRIPTIONS_KEY,
                kind="json",
                description=(
                    "JSON object mapping tool name to a replacement description. "
                    f"Available tools: {names}. Only include tools whose "
                    "description you want to override."
                ),
            )
        )
    return GenomeSpec(components=tuple(components), task=task)


def seed_agent_payload(
    system_prompt: str, *, instructions: str = "", tools: Sequence[AgentTool] = ()
) -> dict[str, str]:
    """A generation-0 payload matching `agent_genome`'s components."""

    payload = {SYSTEM_PROMPT_KEY: system_prompt, INSTRUCTIONS_KEY: instructions}
    if tools:
        payload[TOOL_DESCRIPTIONS_KEY] = "{}"
    return payload


def build_genome_agent(
    candidate: Candidate,
    *,
    provider: Provider,
    tools: Sequence[AgentTool] = (),
    name: str = "candidate_agent",
    target: str = "user",
) -> Agent:
    """Build a live `Agent` from a standard agent-genome candidate.

    The bridge `agent_task_evaluator` needs: system prompt + instructions
    become the effective system prompt; `tool_descriptions` overrides are
    applied with `dataclasses.replace` onto the supplied tools. Unknown tool
    names in the overrides raise (the JSON was valid but meaningless — the
    loop records the candidate as incorrect rather than silently ignoring
    the mutation).
    """

    payload = candidate.payload
    system_prompt = str(payload.get(SYSTEM_PROMPT_KEY, ""))
    instructions = str(payload.get(INSTRUCTIONS_KEY, ""))
    if instructions.strip():
        system_prompt = f"{system_prompt}\n\n{instructions}".strip()

    overrides_json = str(payload.get(TOOL_DESCRIPTIONS_KEY, "") or "{}")
    overrides = json.loads(overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{TOOL_DESCRIPTIONS_KEY} must be a JSON object, got: {overrides_json[:100]}"
        )
    known = {tool.name for tool in tools}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"{TOOL_DESCRIPTIONS_KEY} names unknown tools {unknown}; known: {sorted(known)}"
        )
    effective_tools = tuple(
        dataclasses.replace(tool, description=str(overrides[tool.name]))
        if tool.name in overrides
        else tool
        for tool in tools
    )

    return make_llm_agent(
        name=name,
        provider=provider,
        system_prompt=system_prompt,
        tools=effective_tools,
        target=target,
    )
