#!/usr/bin/env python3

"""Rebuild shorter, denser SWE-bench Pro chains from recorded issue chains.

This reuses the ``files`` already recorded by a previous ``analyze`` run (i.e. the
post LLM + noise filter selection) so it performs **no** GitHub or LLM calls and is
fully deterministic. It then re-links issues with a hot-file cutoff
(``ignore_file_freq``: files touched by more than this many issues no longer act as a
linking bridge) and re-exports the runner manifest, mirroring
``analyze_swebench_pro_chains.py`` + ``export_swebench_pro_chain_experiments.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK_DATA_DIR = ROOT / "datasets" / "swebench_pro"
WORK_CACHE_DIR = WORK_DATA_DIR / "cache"
VENDORED_DATA_DIR = ROOT / "evals" / "swebench" / "data"


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def issues_by_repo(chains_path: Path) -> dict[str, list[dict]]:
    data = json.loads(chains_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for repo in data["repos"]:
        for chain in repo["chains"]:
            for issue in chain["issues"]:
                grouped[repo["repo"]].append(issue)
    return grouped


def link_chains(
    issues: list[dict], min_chain_size: int, ignore_file_freq: int
) -> list[list[dict]]:
    issues = sorted(issues, key=lambda i: (i["commit_time"], i["instance_id"]))
    freq = Counter(file for issue in issues for file in issue["files"])
    union_find = UnionFind(len(issues))
    first_issue_by_file: dict[str, int] = {}
    for idx, issue in enumerate(issues):
        for file in issue["files"]:
            if ignore_file_freq and freq[file] > ignore_file_freq:
                continue
            if file in first_issue_by_file:
                union_find.union(first_issue_by_file[file], idx)
            else:
                first_issue_by_file[file] = idx
    groups: dict[int, list[dict]] = defaultdict(list)
    for idx in range(len(issues)):
        groups[union_find.find(idx)].append(issues[idx])
    chains = [
        sorted(group, key=lambda i: (i["commit_time"], i["instance_id"]))
        for group in groups.values()
    ]
    return sorted(
        (chain for chain in chains if len(chain) >= min_chain_size),
        key=lambda chain: (chain[0]["commit_time"], chain[0]["instance_id"]),
    )


def build_issue_chains(
    grouped: dict[str, list[dict]], min_chain_size: int, ignore_file_freq: int
) -> dict:
    repos = []
    for repo, issues in sorted(grouped.items()):
        sorted_issues = sorted(
            issues, key=lambda i: (i["commit_time"], i["instance_id"])
        )
        chains = []
        for chain_idx, chain_issues in enumerate(
            link_chains(issues, min_chain_size, ignore_file_freq), start=1
        ):
            file_counts = Counter(
                file for issue in chain_issues for file in issue["files"]
            )
            chains.append(
                {
                    "chain_id": f"{repo.replace('/', '__')}-{chain_idx:04d}",
                    "issue_count": len(chain_issues),
                    "shared_files": sorted(
                        file for file, count in file_counts.items() if count > 1
                    ),
                    "files": sorted(file_counts),
                    "issues": chain_issues,
                }
            )
        repos.append(
            {
                "repo": repo,
                "issue_count": len(issues),
                "first_commit_time": sorted_issues[0]["commit_time"],
                "last_commit_time": sorted_issues[-1]["commit_time"],
                "chains": chains,
            }
        )
    return {
        "repo_count": len(repos),
        "issue_count": sum(len(v) for v in grouped.values()),
        "repos": repos,
    }


def parse_jsonish_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return parsed if isinstance(parsed, list) else []


def shared_prior_nodes(issues: list[dict], idx: int) -> list[dict]:
    files = set(issues[idx]["files"])
    predecessors = []
    for prior_idx, prior in enumerate(issues[:idx], start=1):
        overlap = sorted(files & set(prior["files"]))
        if overlap:
            predecessors.append(
                {
                    "step_index": prior_idx,
                    "instance_id": prior["instance_id"],
                    "overlap_files": overlap,
                }
            )
    return predecessors


def build_node(chain_id: str, issues: list[dict], idx: int, original: dict) -> dict:
    issue = issues[idx]
    predecessors = shared_prior_nodes(issues, idx)
    prior_instance_ids = [prior["instance_id"] for prior in issues[:idx]]
    duplicate_times = {
        t: c for t, c in Counter(i["commit_time"] for i in issues).items() if c > 1
    }
    return {
        "run_id": f"{chain_id}__step_{idx + 1:03d}",
        "chain_id": chain_id,
        "memory_session_id": chain_id,
        "step_index": idx + 1,
        "is_chain_seed": idx == 0,
        "repo": original["repo"],
        "repo_language": original.get("repo_language"),
        "instance_id": issue["instance_id"],
        "base_commit": issue["base_commit"],
        "commit_time": issue["commit_time"],
        "dockerhub_tag": original.get("dockerhub_tag"),
        "files_source": issue["files_source"],
        "files": issue["files"],
        "raw_files": issue["raw_files"],
        "prior_instance_ids": prior_instance_ids,
        "memory_available_count": len(prior_instance_ids),
        "direct_predecessors": predecessors,
        "direct_predecessor_count": len(predecessors),
        "has_direct_prior_file_overlap": bool(predecessors),
        "future_bridge_only": idx > 0 and not predecessors,
        "duplicate_time_group_size": duplicate_times.get(issue["commit_time"], 1),
        "chain_length": len(issues),
        "selected_test_files_to_run": parse_jsonish_list(
            original.get("selected_test_files_to_run", "")
        ),
        "fail_to_pass": parse_jsonish_list(original.get("fail_to_pass", "")),
        "pass_to_pass": parse_jsonish_list(original.get("pass_to_pass", "")),
    }


def build_chain(
    repo: str, chain: dict, original_by_id: dict[str, dict], chain_order: int
) -> dict:
    issues = chain["issues"]
    nodes = [
        build_node(chain["chain_id"], issues, idx, original_by_id[issue["instance_id"]])
        for idx, issue in enumerate(issues)
    ]
    return {
        "experiment_id": chain["chain_id"],
        "chain_id": chain["chain_id"],
        "chain_order": chain_order,
        "repo": repo,
        "issue_count": chain["issue_count"],
        "start_commit_time": issues[0]["commit_time"],
        "end_commit_time": issues[-1]["commit_time"],
        "shared_files": chain["shared_files"],
        "all_relation_files": chain["files"],
        "memory_policy": {
            "reset_memory_at_chain_start": True,
            "run_nodes_in_order": True,
            "reuse_chain_memory_for_later_nodes": True,
        },
        "diagnostics": {
            "future_bridge_only_nodes": [
                node["instance_id"] for node in nodes if node["future_bridge_only"]
            ],
            "duplicate_time_nodes": [
                node["instance_id"]
                for node in nodes
                if node["duplicate_time_group_size"] > 1
            ],
        },
        "nodes": nodes,
    }


def rebuild_manifests(
    chains_path: Path,
    dataset_path: Path,
    ignore_file_freq: int,
    min_chain_size: int,
    out_chains: Path,
    out_experiments: Path,
    out_nodes: Path,
) -> dict:
    """Relink recorded issues, export all outputs, and return the manifest."""

    result = build_issue_chains(
        issues_by_repo(chains_path), min_chain_size, ignore_file_freq
    )
    original_by_id = {
        item["instance_id"]: item
        for item in json.loads(dataset_path.read_text(encoding="utf-8"))
    }
    chains = [
        build_chain(repo["repo"], chain, original_by_id, chain_order)
        for chain_order, (repo, chain) in enumerate(
            ((repo, chain) for repo in result["repos"] for chain in repo["chains"]),
            start=1,
        )
    ]
    manifest = {
        "schema_version": 1,
        "source_chains": str(chains_path),
        "source_dataset": str(dataset_path),
        "ignore_file_freq": ignore_file_freq,
        "min_chain_size": min_chain_size,
        "experiment_count": len(chains),
        "node_count": sum(chain["issue_count"] for chain in chains),
        "chains": chains,
    }
    for path in (out_chains, out_experiments, out_nodes):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_chains.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out_experiments.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    out_nodes.write_text(
        "".join(json.dumps(node) + "\n" for chain in chains for node in chain["nodes"]),
        encoding="utf-8",
    )

    lengths = sorted((chain["issue_count"] for chain in chains), reverse=True)
    print(
        f"chains={len(lengths)} chained_issues={sum(lengths)} max={lengths[0] if lengths else 0} "
        f">=20:{sum(length >= 20 for length in lengths)} "
        f"10-19:{sum(10 <= length < 20 for length in lengths)} "
        f"4-9:{sum(4 <= length < 10 for length in lengths)}"
    )
    print(f"lengths={lengths}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chains-path",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_issue_chains.json",
        help="Recorded issue-chain JSON, including files and commit_time.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=WORK_DATA_DIR / "swe_bench_pro.json",
        help="Local JSON export of the ScaleAI/SWE-bench_Pro test split.",
    )
    parser.add_argument(
        "--ignore-file-freq",
        type=int,
        default=4,
        help="Ignore files touched by more than N issues when linking.",
    )
    parser.add_argument(
        "--min-chain-size",
        type=int,
        default=3,
        help="Minimum issue count required for a deep chain.",
    )
    parser.add_argument(
        "--out-chains",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_issue_chains_deep.json",
        help="Output relinked issue-chain JSON.",
    )
    parser.add_argument(
        "--out-experiments",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_chain_experiments_deep.json",
        help="Output grouped experiment manifest.",
    )
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=VENDORED_DATA_DIR / "swe_bench_pro_chain_experiment_nodes_deep.jsonl",
        help="Output one-node-per-line runner manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rebuild_manifests(
        args.chains_path,
        args.dataset_path,
        args.ignore_file_freq,
        args.min_chain_size,
        args.out_chains,
        args.out_experiments,
        args.out_nodes,
    )


if __name__ == "__main__":
    main()
