"""LLM-driven proposal: render parents into a prompt, parse a payload back.

`llm_propose` is a helper `ProposeFn`, not the harness core — a scripted
mutation function satisfies the same seam. It edits only the named string
`fields` of the payload; everything else is carried over from the parent
unchanged, so structured/non-text payload entries survive LLM mutation.

The wire format is deliberately dumb and greppable: each editable field goes
out (and must come back) as

    ### <field>
    ```
    <content>
    ```

The model call itself hides behind `ask: Callable[[str], str]`, so tests and
offline demos inject a function and never touch a provider;
`ask_from_provider` adapts a real `llm.Provider` (with the shared throttling
retry) for live runs.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Mapping, Sequence

from simple_agent_lab.llm import (
    LLMRequest,
    Provider,
    complete_with_retry,
    llm_message,
)
from simple_agent_lab.messages import text_of

from .types import EvolutionRecord, Proposal, ProposeFn

AskFn = Callable[[str], str]

# Longest inspiration/feedback excerpts shown to the model; parents are shown
# in full because they are what gets edited.
_INSPIRATION_CHARS = 1500
_FEEDBACK_CHARS = 2000
_NOTE_CHARS = 500


def ask_from_provider(
    provider: Provider, *, system_prompt: str = "", max_tokens: int | None = None
) -> AskFn:
    """A plain text-in/text-out call on `provider` (throttling retried)."""

    def ask(prompt: str) -> str:
        request = LLMRequest(
            provider=provider,
            messages=[llm_message("user", prompt)],
            system_prompt=system_prompt or None,
            max_tokens=max_tokens,
        )
        return text_of(complete_with_retry(request).content)

    return ask


def render_fields(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Render payload `fields` in the fenced wire format shown to the model."""

    return "\n\n".join(f"### {field}\n```\n{payload[field]}\n```" for field in fields)


def parse_fields(response: str, fields: Sequence[str]) -> dict[str, str]:
    """Extract `### field` fenced blocks from a model response.

    Returns only the fields present; a proposer decides whether missing
    fields inherit from the parent. Raises `ValueError` when the response
    contains none of the requested fields (an unusable proposal — the loop
    records it as a failure instead of silently re-evaluating the parent).
    """

    found: dict[str, str] = {}
    for field in fields:
        pattern = re.compile(
            rf"^###\s*{re.escape(field)}\s*\n```[^\n]*\n(.*?)\n?```",
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(response)
        if match:
            found[field] = match.group(1)
    if not found:
        raise ValueError(
            f"proposal contained none of the editable fields {list(fields)}; "
            f"response started: {response[:200]!r}"
        )
    return found


def _proposal_note(response: str) -> str:
    """The model's rationale: the response with fenced blocks stripped."""

    without_blocks = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
    without_headers = re.sub(r"^###.*$", "", without_blocks, flags=re.MULTILINE)
    return " ".join(without_headers.split())[:_NOTE_CHARS]


def build_mutation_prompt(
    parents: Sequence[EvolutionRecord],
    *,
    task: str,
    fields: Sequence[str],
    guidance: str = "",
) -> str:
    """One prompt: task, the parent (full), inspirations (excerpted), feedback.

    Exposed separately so experiments can inspect or unit-test exactly what
    the model sees — the prompt is part of the method, not harness magic.
    """

    parent = parents[0]
    lines = [
        "You are improving a candidate in an evolutionary search.",
        "",
        f"Goal: {task}",
    ]
    if guidance:
        lines += ["", f"Guidance: {guidance}"]
    lines += [
        "",
        f"Current candidate (fitness {parent.evaluation.fitness:.4f}):",
        render_fields(dict(parent.candidate.payload), fields),
    ]
    if parent.evaluation.feedback:
        lines += [
            "",
            "Evaluator feedback on the current candidate:",
            parent.evaluation.feedback[:_FEEDBACK_CHARS],
        ]
    for inspiration in parents[1:]:
        lines += [
            "",
            f"For inspiration, another strong candidate "
            f"(fitness {inspiration.evaluation.fitness:.4f}):",
            render_fields(dict(inspiration.candidate.payload), fields)[
                :_INSPIRATION_CHARS
            ],
        ]
    lines += [
        "",
        "Propose ONE improved variant. First explain the change in a sentence "
        "or two, then return every field you changed exactly in this format "
        "(unchanged fields may be omitted):",
        "",
        "### <field name>",
        "```",
        "<new content>",
        "```",
    ]
    return "\n".join(lines)


def llm_propose(
    ask: AskFn,
    *,
    task: str,
    fields: Sequence[str],
    guidance: str = "",
    operator: str = "llm_mutate",
) -> ProposeFn:
    """A `ProposeFn` that asks a model to rewrite the payload's `fields`.

    Fields the response omits are inherited from the parent; payload keys
    outside `fields` are always inherited. The model's prose (minus the
    fenced blocks) becomes the candidate's `note`.
    """

    def propose(parents: Sequence[EvolutionRecord], rng: random.Random) -> Proposal:
        prompt = build_mutation_prompt(
            parents, task=task, fields=fields, guidance=guidance
        )
        response = ask(prompt)
        changed = parse_fields(response, fields)
        payload = dict(parents[0].candidate.payload)
        payload.update(changed)
        return Proposal(
            payload=payload, operator=operator, note=_proposal_note(response)
        )

    return propose
