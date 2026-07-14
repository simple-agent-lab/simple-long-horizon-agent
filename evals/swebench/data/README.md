# Vendored SWE-bench Pro chain manifests

These files pin the *issue chains* used by the memory-chain experiment
(`runs/swebench/run_swebench_pro_memory_chains.py`) so runs use a stable,
repository-owned manifest. They describe **only how instances are grouped and
ordered into chains** — the SWE-bench Pro instances themselves still come from
the `ScaleAI/SWE-bench_Pro` dataset at run time.

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

## Generation code

The code that produces these manifests lives in this repository:

| Script | Role |
| --- | --- |
| [`analyze_swebench_pro_chains.py`](../../../scripts/swebench/analyze_swebench_pro_chains.py) | Extract relation files, resolve commit times, filter noise, and form chronological per-repository issue chains. |
| [`export_swebench_pro_chain_experiments.py`](../../../scripts/swebench/export_swebench_pro_chain_experiments.py) | Add runner fields and export the ordinary flat JSONL manifest. |
| [`rebuild_deep_chains.py`](../../../scripts/swebench/rebuild_deep_chains.py) | Deterministically relink a recorded analysis with a hot-file cutoff and export the deep manifest. |

`render_swebench_pro_chains.py` only rendered an optional HTML inspection
report; it was not part of either JSONL generation path and is not needed here.

## Inputs and reproducibility boundary

The scripts keep large or provider-derived working inputs under the gitignored
`datasets/` tree. A full refresh needs:

- a JSON-array export of the `ScaleAI/SWE-bench_Pro` test split;
- GitHub commit timestamps, fetched by the analyzer and cached locally;
- for patch-fallback rows, either the recorded LLM file-selection cache or an
  OpenAI-compatible model configured through the analyzer flags.

The analysis stage can therefore make GitHub and model calls. Once its
issue-chain JSON has been recorded, export and deep relinking are deterministic.
The deep rebuild was checked against the vendored deep JSONL using the recorded
analysis inputs: both files have SHA-256
`a4c52ec33767aaa93ccf80d912a22d0160c6ffde6c89c7fed667e85df364c85d`.

The ordinary JSONL predates the retained analysis cache. Its generator is
preserved, but byte-for-byte regeneration requires the original analysis
snapshot; a fresh model-backed analysis may select different relation files.

## Refresh workflow

Run every command from the repository root. First export the upstream dataset
to the ignored working directory:

```bash
uv run --extra swebench python - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

path = Path("datasets/swebench_pro/swe_bench_pro.json")
path.parent.mkdir(parents=True, exist_ok=True)
rows = [dict(row) for row in load_dataset("ScaleAI/SWE-bench_Pro", split="test")]
path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
print(f"wrote {path} ({len(rows)} rows)")
PY
```

Build and export the ordinary chain analysis (minimum chain size 4):

```bash
uv run python scripts/swebench/analyze_swebench_pro_chains.py \
  --min-chain-size 4 \
  --output-path datasets/swebench_pro/cache/swe_bench_pro_issue_chains_standard.json

uv run python scripts/swebench/export_swebench_pro_chain_experiments.py \
  --chains-path datasets/swebench_pro/cache/swe_bench_pro_issue_chains_standard.json
```

For the deep analysis, retain all initial components, then relink with the
recorded experiment settings (`ignore_file_freq=4`, `min_chain_size=3`):

```bash
uv run python scripts/swebench/analyze_swebench_pro_chains.py \
  --min-chain-size 1 \
  --output-path datasets/swebench_pro/cache/swe_bench_pro_issue_chains.json

uv run python scripts/swebench/rebuild_deep_chains.py \
  --ignore-file-freq 4 \
  --min-chain-size 3
```

Use `--offline` when the commit-time cache is complete. Use
`--no-llm-noise-filter` for a model-free analysis with different semantics.
`instance_id` values must match the dataset split; a mismatch surfaces as
`missing_instance_ids` in the run's `experiment.json`.

The focused deterministic check is:

```bash
uv run python -m unittest tests.unit.test_swebench_chain_data -v
```
