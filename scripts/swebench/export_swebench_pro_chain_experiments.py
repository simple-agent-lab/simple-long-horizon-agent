#!/usr/bin/env python3

"""Export analyzed SWE-bench Pro chains into runner-friendly manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK_DATA_DIR = ROOT / "datasets" / "swebench_pro"
WORK_CACHE_DIR = WORK_DATA_DIR / "cache"
VENDORED_DATA_DIR = ROOT / "evals" / "swebench" / "data"


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


def duplicate_time_groups(issues: list[dict]) -> dict[str, int]:
    return {
        time: count
        for time, count in Counter(issue["commit_time"] for issue in issues).items()
        if count > 1
    }


def build_node(chain: dict, issues: list[dict], idx: int, original: dict) -> dict:
    issue = issues[idx]
    predecessors = shared_prior_nodes(issues, idx)
    prior_instance_ids = [prior["instance_id"] for prior in issues[:idx]]
    duplicate_times = duplicate_time_groups(issues)
    return {
        "run_id": f"{chain['chain_id']}__step_{idx + 1:03d}",
        "chain_id": chain["chain_id"],
        "memory_session_id": chain["chain_id"],
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


def parse_jsonish_list(value: str) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_chain(
    repo: str, chain: dict, original_by_id: dict[str, dict], chain_order: int
) -> dict:
    issues = chain["issues"]
    nodes = [
        build_node(chain, issues, idx, original_by_id[issue["instance_id"]])
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


def export_manifests(
    chains_path: Path,
    dataset_path: Path,
    output_json: Path,
    output_jsonl: Path,
) -> dict:
    """Export grouped and flat manifests and return the grouped value."""

    result = json.loads(chains_path.read_text(encoding="utf-8"))
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
        "experiment_count": len(chains),
        "node_count": sum(chain["issue_count"] for chain in chains),
        "chains": chains,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(node) + "\n" for chain in chains for node in chain["nodes"]),
        encoding="utf-8",
    )
    print(f"wrote {output_json} and {output_jsonl} ({len(chains)} chains)")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chains-path",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_issue_chains.json",
        help="Input issue-chain analysis JSON.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=WORK_DATA_DIR / "swe_bench_pro.json",
        help="Local JSON export of the ScaleAI/SWE-bench_Pro test split.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_chain_experiments.json",
        help="Grouped experiment manifest.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=VENDORED_DATA_DIR / "swe_bench_pro_chain_experiment_nodes.jsonl",
        help="One-node-per-line runner manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_manifests(
        args.chains_path,
        args.dataset_path,
        args.output_json,
        args.output_jsonl,
    )


if __name__ == "__main__":
    main()
