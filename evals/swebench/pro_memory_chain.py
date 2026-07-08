"""Memory-chain SWE-bench Pro experiment helpers.

This is the pure planning/configuration half of the *memory-based* repo-chain
runner. It is a peer of ``pro_repo_chain`` but with a different research shape,
borrowed from the ``mini-memory`` filesystem-memory chains:

- Chains come from a pre-analyzed *issue-chains* JSON (repos -> chains ->
  issues), not from splitting a repo by commit time.
- Each issue in a chain runs as an ordinary, isolated SWE-bench Pro instance in
  a *fresh* agent context. Nothing about the previous instance's transcript is
  carried forward in-context. The only thing that crosses instance boundaries is
  Simple Agent Lab filesystem memory, scoped per chain
  (``SAL_MEMORY_NAME=<chain_id>``): the model reads the chain's memory dir at the
  start of each instance and the run-end distiller updates it, so later issues in
  the chain can reuse earlier lessons.
- The full SWE-bench Pro split still runs: every dataset instance not covered by
  a chain becomes a length-1 singleton run unit (memory off by default).
- Run units are ordered longest-first so a fixed parallel pool starts the long
  chains early and the many short/singleton runs fill the remaining lanes while
  those long chains keep occupying theirs.

The executable runner (``runs/swebench/run_swebench_pro_memory_chains.py``)
builds on these helpers so tests can lock the research contract without starting
Docker or calling a model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"
# The analyzed issue-chains file lives outside this repo (produced by the
# mini-memory chain analysis). It is only a convenience default for
# ``--chains-json``; the runner errors clearly if the path is missing.
DEFAULT_CHAINS_JSON = "~/code/mini-memory/data/swe_bench_pro_issue_chains_deep.json"
DEFAULT_API_KIND = "openai-responses"
DEFAULT_MAX_TURNS = 250
DEFAULT_AGENT_FLAVOR = "bash"
# Simple flavors are built through the generic runner's ``agent_spec`` path,
# which is the path that also installs the filesystem-memory lifecycle hooks.
# Workflow arms (loop / goal / pdr) use a custom ``build_agent`` facade that does
# not wire memory, so they are intentionally excluded from the memory chain.
MEMORY_CHAIN_AGENT_FLAVORS = ("bash", "bash_task", "bash_task_read", "bash_skills")

SINGLETON_CHAIN_SOURCE = "singleton"
CHAIN_SOURCE = "chain"


@dataclass(frozen=True)
class MemoryChain:
    """One ordered run unit: a memory-sharing chain or a lone singleton.

    ``rows`` are the SWE-bench Pro instance dicts in execution order. A chain
    (``source == "chain"``) shares one filesystem-memory namespace across its
    instances; a singleton (``source == "singleton"``) is a single instance that
    was not part of any analyzed chain.
    """

    chain_id: str
    repo: str
    rows: tuple[dict[str, Any], ...]
    memory_enabled: bool
    source: str = CHAIN_SOURCE

    @property
    def length(self) -> int:
        return len(self.rows)

    @property
    def is_singleton(self) -> bool:
        return self.source == SINGLETON_CHAIN_SOURCE

    @property
    def instance_ids(self) -> list[str]:
        return [str(row.get("instance_id") or "") for row in self.rows]


@dataclass(frozen=True)
class RawIssueChain:
    """A chain as declared by the issue-chains JSON, before dataset matching."""

    chain_id: str
    repo: str
    instance_ids: tuple[str, ...]


def load_issue_chains(path: str | Path) -> list[RawIssueChain]:
    """Parse an issue-chains JSON into commit-time-ordered raw chains.

    The expected shape is ``{"repos": [{"repo", "chains": [{"chain_id",
    "issues": [{"instance_id", "commit_time"}]}]}]}`` — the structure produced by
    the mini-memory chain analysis. Issues inside a chain are re-sorted by
    ``(commit_time, instance_id)`` defensively so execution order stays
    chronological even if the file is out of order.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return chains_from_manifest(data)


def chains_from_manifest(data: Mapping[str, Any]) -> list[RawIssueChain]:
    """Build raw chains from an already-loaded issue-chains manifest mapping."""

    chains: list[RawIssueChain] = []
    for repo_entry in data.get("repos", []) or []:
        if not isinstance(repo_entry, Mapping):
            continue
        repo = str(repo_entry.get("repo") or "").strip()
        for chain in repo_entry.get("chains", []) or []:
            if not isinstance(chain, Mapping):
                continue
            issues = [
                issue
                for issue in (chain.get("issues", []) or [])
                if isinstance(issue, Mapping) and issue.get("instance_id")
            ]
            ordered = sorted(
                issues,
                key=lambda issue: (
                    str(issue.get("commit_time") or ""),
                    str(issue.get("instance_id") or ""),
                ),
            )
            instance_ids = tuple(str(issue.get("instance_id")) for issue in ordered)
            if not instance_ids:
                continue
            chains.append(
                RawIssueChain(
                    chain_id=str(chain.get("chain_id") or "").strip()
                    or f"{repo}-{instance_ids[0]}",
                    repo=repo,
                    instance_ids=instance_ids,
                )
            )
    return chains


@dataclass(frozen=True)
class MemoryChainPlan:
    """The ordered run plan plus the bookkeeping needed for the manifest."""

    chains: tuple[MemoryChain, ...]
    missing_instance_ids: tuple[str, ...]
    duplicate_instance_ids: tuple[str, ...]

    @property
    def chain_units(self) -> list[MemoryChain]:
        return [chain for chain in self.chains if not chain.is_singleton]

    @property
    def singleton_units(self) -> list[MemoryChain]:
        return [chain for chain in self.chains if chain.is_singleton]

    @property
    def instance_count(self) -> int:
        return sum(chain.length for chain in self.chains)


def plan_memory_chains(
    rows: Sequence[Mapping[str, Any]],
    raw_chains: Sequence[RawIssueChain],
    *,
    memory: bool = True,
    singleton_memory: bool = False,
) -> MemoryChainPlan:
    """Turn dataset rows + analyzed chains into an ordered list of run units.

    Every dataset instance is placed exactly once: instances named by a chain
    keep that chain's order and share its memory namespace; every remaining
    instance becomes a length-1 singleton. Chain instances missing from the
    dataset are reported (not fatal) so a mismatched dataset/chain file surfaces
    loudly. The result is ordered longest-first.
    """

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id and instance_id not in by_id:
            by_id[instance_id] = dict(row)

    used: set[str] = set()
    missing: list[str] = []
    duplicate: list[str] = []
    chains: list[MemoryChain] = []

    for raw in raw_chains:
        chain_rows: list[dict[str, Any]] = []
        for instance_id in raw.instance_ids:
            if instance_id not in by_id:
                missing.append(instance_id)
                continue
            if instance_id in used:
                duplicate.append(instance_id)
                continue
            used.add(instance_id)
            chain_rows.append(by_id[instance_id])
        if not chain_rows:
            continue
        chains.append(
            MemoryChain(
                chain_id=raw.chain_id,
                repo=raw.repo or str(chain_rows[0].get("repo") or ""),
                rows=tuple(chain_rows),
                memory_enabled=bool(memory),
                source=CHAIN_SOURCE,
            )
        )

    singleton_rows = [row for iid, row in by_id.items() if iid not in used]
    singleton_rows.sort(
        key=lambda row: (
            str(row.get("repo") or ""),
            str(row.get("base_commit") or ""),
            str(row.get("instance_id") or ""),
        )
    )
    for row in singleton_rows:
        chains.append(
            MemoryChain(
                chain_id=singleton_chain_id(row),
                repo=str(row.get("repo") or ""),
                rows=(row,),
                memory_enabled=bool(memory and singleton_memory),
                source=SINGLETON_CHAIN_SOURCE,
            )
        )

    return MemoryChainPlan(
        chains=tuple(order_chains_longest_first(chains)),
        missing_instance_ids=tuple(missing),
        duplicate_instance_ids=tuple(duplicate),
    )


def order_chains_longest_first(chains: Iterable[MemoryChain]) -> list[MemoryChain]:
    """Order run units so the longest chains are submitted first.

    Longer chains occupy a parallel lane for longer, so starting them first lets
    the shorter chains and singletons fill the remaining lanes instead of leaving
    a long chain queued behind a pile of quick singletons. Within one length,
    real chains sort ahead of singletons (a same-length chain still carries
    memory worth starting first); remaining ties are broken deterministically by
    repo then chain id.
    """

    return sorted(
        chains,
        key=lambda chain: (
            -chain.length,
            chain.is_singleton,
            chain.repo,
            chain.chain_id,
        ),
    )


def singleton_chain_id(row: Mapping[str, Any]) -> str:
    """Stable run-unit id for an instance that is not part of any chain."""

    return str(row.get("instance_id") or "").strip() or "singleton"


def expand_auth_slots(spec: str | None, *, default_env: str) -> list[str]:
    """Expand an ``ENV:COUNT,ENV2:COUNT2`` spec into one env name per slot.

    Unlike the repo-chain runner's variant, this does not pad to a chain count:
    it returns exactly the declared slots, and the runner sizes the concurrency
    pool from the length of this list. An empty spec means a single slot on the
    default auth env.
    """

    text = (spec or "").strip()
    if not text:
        return [default_env]
    expanded: list[str] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("provider auth spec contains an empty entry")
        auth_env, separator, raw_count = part.partition(":")
        auth_env = auth_env.strip()
        if not is_env_var_name(auth_env):
            raise ValueError(f"invalid auth env var name {auth_env!r}")
        if not separator:
            raise ValueError(f"auth slots must use ENV:COUNT, got {part!r}")
        try:
            count = int(raw_count.strip())
        except ValueError:
            raise ValueError(
                f"auth slot count for {auth_env} must be an integer"
            ) from None
        if count <= 0:
            raise ValueError(f"auth slot count for {auth_env} must be positive")
        expanded.extend([auth_env] * count)
    return expanded


def lane_auth_slots(expanded: Sequence[str], parallel: int) -> list[str]:
    """Assign an auth env to each of ``parallel`` concurrent lanes.

    Concurrency is bounded by the pool, and each lane holds one auth slot for the
    whole chain it is running. When there are more lanes than declared slots the
    slots cycle; when there are fewer, only the first lanes' slots are used.
    """

    if parallel <= 0:
        raise ValueError("parallel must be positive")
    if not expanded:
        raise ValueError("need at least one auth slot")
    return [expanded[index % len(expanded)] for index in range(parallel)]


def is_env_var_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(char.isalnum() or char == "_" for char in value)


@dataclass(frozen=True)
class ProMemoryChainConfig:
    """Operator-visible configuration for the Pro memory-chain experiment."""

    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    model: str = ""
    api_kind: str = DEFAULT_API_KIND
    reasoning_effort: str = ""
    max_turns: int = DEFAULT_MAX_TURNS
    agent_flavor: str = DEFAULT_AGENT_FLAVOR
    task_tool: bool = False
    memory: bool = True
    singleton_memory: bool = False
    model_name: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "model": self.model,
            "api_kind": self.api_kind,
            "reasoning_effort": self.reasoning_effort,
            "max_turns": self.max_turns,
            "agent_flavor": self.agent_flavor,
            "task_tool": self.task_tool,
            "memory": self.memory,
            "singleton_memory": self.singleton_memory,
            "model_name": self.model_name,
        }


def model_name_for_config(
    *, agent_flavor: str, memory: bool, singleton_memory: bool
) -> str:
    """Prediction ``model_name`` describing the memory-chain arm."""

    if not memory:
        memory_tag = "nomemory"
    elif singleton_memory:
        memory_tag = "memory-all"
    else:
        memory_tag = "memory"
    return f"simple-agent-lab-pro-memory-chain-{agent_flavor}-{memory_tag}"


def plan_manifest(
    plan: MemoryChainPlan,
    *,
    config: ProMemoryChainConfig,
    run_id: str,
    parallel: int,
) -> dict[str, Any]:
    """A JSON-friendly manifest capturing the run plan and match statistics."""

    per_repo: dict[str, dict[str, int]] = defaultdict(
        lambda: {"chains": 0, "chain_instances": 0, "singletons": 0}
    )
    for chain in plan.chains:
        bucket = per_repo[chain.repo]
        if chain.is_singleton:
            bucket["singletons"] += 1
        else:
            bucket["chains"] += 1
            bucket["chain_instances"] += chain.length

    length_histogram: dict[int, int] = defaultdict(int)
    for chain in plan.chain_units:
        length_histogram[chain.length] += 1

    return {
        "schema": "simple-agent-lab.swebench-pro-memory-chain-experiment.v1",
        "run_id": run_id,
        "config": config.as_record(),
        "parallel": parallel,
        "run_unit_count": len(plan.chains),
        "chain_count": len(plan.chain_units),
        "singleton_count": len(plan.singleton_units),
        "instance_count": plan.instance_count,
        "chain_instance_count": sum(chain.length for chain in plan.chain_units),
        "missing_instance_ids": list(plan.missing_instance_ids),
        "duplicate_instance_ids": list(plan.duplicate_instance_ids),
        "chain_length_histogram": {
            str(length): count for length, count in sorted(length_histogram.items())
        },
        "per_repo": {repo: dict(counts) for repo, counts in sorted(per_repo.items())},
        "order": [
            {
                "chain_id": chain.chain_id,
                "repo": chain.repo,
                "length": chain.length,
                "source": chain.source,
                "memory_enabled": chain.memory_enabled,
                "instance_ids": chain.instance_ids,
            }
            for chain in plan.chains
        ],
    }
