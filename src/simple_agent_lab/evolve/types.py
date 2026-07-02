"""Data values and callable seams for the evolution harness.

The whole harness is one loop (`loop.run_evolution`) over four seams, each a
plain callable — mirroring how `core.GenerateFn` keeps the agent loop free of
class hierarchies:

- `ProposeFn`   — how a new candidate payload is produced (LLM or scripted).
- `EvaluateFn`  — the feedback signal for one candidate.
- `SelectFn`    — parent + inspiration sampling from the archive.
- `AcceptFn`    — the explicit accept/reject decision, with a reason.

What evolves is *data*: `Candidate.payload` is a JSON-able mapping (prompt
text, program source, tool descriptions, a config...). The harness never
interprets the payload; only the user's proposer and evaluator do. That one
choice is what lets the same loop cover prompt evolution, harness evolution
(arXiv:2604.25850), and ShinkaEvolve-style program evolution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from .archive import Archive

# What evolves. Must stay JSON-serializable so the archive can persist it.
Payload = Mapping[str, Any]


@dataclass(frozen=True)
class Candidate:
    """One point in the search space: a payload plus its lineage.

    `operator` names what produced it (`"seed"`, a proposer's label like
    `"mutate"`, or `"propose_error"` when proposing itself failed). `note` is
    the proposer's free-text rationale — for an LLM proposer, the model's own
    explanation of the change — kept so a reader of the archive can see *why*
    each candidate was tried, not just what it contained.
    """

    id: str
    payload: Payload
    parent_ids: tuple[str, ...] = ()
    operator: str = "seed"
    generation: int = 0
    note: str = ""


@dataclass(frozen=True)
class Evaluation:
    """The feedback signal for one candidate.

    `fitness` is the scalar being maximized. `correct=False` marks a candidate
    that failed evaluation outright (crashed, invalid, broke a constraint):
    it is recorded for the audit trail but never selectable as a parent.
    `metrics` keeps raw structured values; `feedback` is free text (failing
    cases, error output) that a proposer can feed back into the next prompt.
    """

    fitness: float = 0.0
    correct: bool = True
    metrics: Mapping[str, Any] = field(default_factory=dict)
    feedback: str = ""
    error: str = ""


@dataclass(frozen=True)
class Decision:
    """An acceptance verdict. `reason` is mandatory context, not decoration:

    the archive record must let another person or agent see why a candidate
    was kept or dropped (ADR treat-self-evolution-as-harness-capability).
    """

    accepted: bool
    reason: str


@dataclass(frozen=True)
class EvolutionRecord:
    """One archive line: candidate + evaluation + decision. Append-only."""

    candidate: Candidate
    evaluation: Evaluation
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Proposal:
    """What a `ProposeFn` returns: the new payload plus provenance labels."""

    payload: Payload
    operator: str = "mutate"
    note: str = ""


# The four seams. `parents` is non-empty; parents[0] is the parent proper and
# the rest are inspirations (context the proposer may draw on).
ProposeFn = Callable[[Sequence[EvolutionRecord], random.Random], Proposal]
EvaluateFn = Callable[[Candidate], Evaluation]
SelectFn = Callable[["Archive", random.Random], Sequence[EvolutionRecord]]
AcceptFn = Callable[[Candidate, Evaluation, "Archive"], Decision]
