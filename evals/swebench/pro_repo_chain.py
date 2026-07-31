"""Small planning/configuration values for SWE-bench Pro repo chains."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from simple_long_horizon_agent.messages import MessageKind

DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"
DEFAULT_API_KIND = "openai-responses"
DEFAULT_CONTEXT_WINDOW_TOKENS = 272_000
DEFAULT_THRESHOLD_TOKENS = int(DEFAULT_CONTEXT_WINDOW_TOKENS * 0.8)
DEFAULT_KEEP_RECENT = 4
DEFAULT_PRESERVE_KINDS: tuple[MessageKind, ...] = (
    "task",
    "system",
    "context",
)
DEFAULT_MODEL_NAME = "simple-long-horizon-agent-pro-repo-chain-bash-none"


def group_instances_by_repo(
    instances: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        repo = str(row.get("repo") or "").strip() or "unknown"
        groups[repo].append(dict(row))
    return dict(groups)


@dataclass(frozen=True)
class ProRepoExperimentConfig:
    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    model: str = ""
    api_kind: str = DEFAULT_API_KIND
    reasoning_effort: str = ""
    max_turns: int = 250
    context_window_tokens: int = DEFAULT_THRESHOLD_TOKENS
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS
    keep_recent: int = DEFAULT_KEEP_RECENT
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS
    agent_flavor: str = "bash"
    task_tool: bool = False
    compression_strategy: str = "none"
    handoff: bool = True
    model_name: str = DEFAULT_MODEL_NAME

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "model": self.model,
            "api_kind": self.api_kind,
            "reasoning_effort": self.reasoning_effort,
            "max_turns": self.max_turns,
            "context_window_tokens": self.context_window_tokens,
            "threshold_tokens": self.threshold_tokens,
            "keep_recent": self.keep_recent,
            "preserve_kinds": list(self.preserve_kinds),
            "agent_flavor": self.agent_flavor,
            "task_tool": self.task_tool,
            "compression_strategy": self.compression_strategy,
            "handoff": self.handoff,
            "model_name": self.model_name,
        }
