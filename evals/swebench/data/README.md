# Vendored SWE-bench Pro chain manifests

These files pin the *issue chains* used by the memory-chain experiment
(`runs/swebench/run_swebench_pro_memory_chains.py`) so a run does not depend on
an external `mini-memory` checkout. They describe **only how instances are
grouped and ordered into chains** — the SWE-bench Pro instances themselves still
come from the `ScaleAI/SWE-bench_Pro` dataset at run time.

## Files

| File | Shape | Contents |
| --- | --- | --- |
| `swe_bench_pro_chain_experiment_nodes_deep.jsonl` | flat JSONL, one node per line | The recommended explicit `--chains-json` for deep-chain experiments (`min_chain_size=3`): 261 in-chain instances across 47 chains. |
| `swe_bench_pro_chain_experiment_nodes.jsonl` | flat JSONL, one node per line | The non-deep chain analysis (28 chains, 347 in-chain instances). Kept for comparison; select it via `--chains-json`. |

Each line is a JSON object. The fields the loader needs are `chain_id`,
`step_index`, `instance_id`, and `repo` (`commit_time` is used as a tiebreak).
The deep file carries extra analysis fields (`files`, `prior_instance_ids`,
`fail_to_pass`, `dockerhub_tag`, …) that the runner ignores.

## How they are consumed

`evals/swebench/pro_memory_chain.py::load_issue_chains` auto-detects the `.jsonl`
suffix and groups nodes by `chain_id`, ordering each chain by `step_index`. Every
dataset instance not named by a chain becomes a length-1 singleton, so the full
731-instance split still runs; chains are then scheduled longest-first. The
loader also still accepts the older nested issue-chains JSON
(`{"repos": [{"chains": [{"issues": [...]}]}]}`) for external manifests.

## Source / refresh

Produced by the `mini-memory` chain analysis over `ScaleAI/SWE-bench_Pro`: its
rebuild script feeds its SWE-bench Pro chain-export script. To refresh,
regenerate the JSONL there and copy it back into this directory. `instance_id`s
must match the dataset split; a mismatch surfaces as `missing_instance_ids` in
the run's `experiment.json`.
