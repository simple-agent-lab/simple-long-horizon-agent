# Vendored SWE-bench Pro chain manifests

These files pin the issue grouping and order used by the SWE-bench Pro chain
runners. The instances themselves still come from the
`ScaleAI/SWE-bench_Pro` dataset at run time.

## Files

| File | Shape | Contents |
| --- | --- | --- |
| `swe_bench_pro_chain_experiment_nodes_deep.jsonl` | flat JSONL, one node per line | The recommended explicit `--chains-json` for deep-chain experiments (`min_chain_size=3`): 261 in-chain instances across 47 chains. |
| `swe_bench_pro_chain_experiment_nodes.jsonl` | flat JSONL, one node per line | The non-deep chain analysis (28 chains, 347 in-chain instances). Kept for comparison; select it via `--chains-json`. |

Each line is a JSON object. The fields the loader needs are `chain_id`,
`step_index`, `instance_id`, and `repo` (`commit_time` is used as a tiebreak).
Analysis-only fields stay in the gitignored analysis JSON rather than being
copied into these runner manifests.

## How they are consumed

`evals/swebench/pro_memory_chain.py::load_issue_chains` groups nodes by
`chain_id`, ordering each chain by `step_index`. Every
dataset instance not named by a chain becomes a length-1 singleton, so the full
731-instance split still runs; chains are then scheduled longest-first. Flat
nodes must carry a non-empty `chain_id`; duplicate IDs and IDs spanning
multiple repos are rejected before planning so memory namespaces and summary
artifacts cannot collide.

## Generation code

[`analyze_swebench_pro_chains.py`](../../../scripts/swebench/analyze_swebench_pro_chains.py)
is the single generation entry point. It extracts relation files, resolves
commit times, filters noise, groups issues, writes the detailed local analysis
JSON, and optionally writes the minimal runner JSONL with
`--nodes-output-path`.

## Inputs and reproducibility boundary

The analyzer keeps large or provider-derived working inputs under the gitignored
`datasets/` tree. A full refresh needs:

- a JSON-array export of the `ScaleAI/SWE-bench_Pro` test split;
- GitHub commit timestamps, fetched by the analyzer and cached locally;
- for patch-fallback rows, either the recorded LLM file-selection cache or an
  OpenAI-compatible model configured through the analyzer flags.

The analyzer can therefore make GitHub and model calls. Cached commit times and
LLM selections make repeated runs stable; a fresh model-backed analysis may
select different relation files.

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

Build the ordinary chain analysis (minimum chain size 4) and its runner
manifest:

```bash
uv run python scripts/swebench/analyze_swebench_pro_chains.py \
  --min-chain-size 4 \
  --output-path datasets/swebench_pro/cache/swe_bench_pro_issue_chains_standard.json \
  --nodes-output-path evals/swebench/data/swe_bench_pro_chain_experiment_nodes.jsonl
```

Build the deep manifest by applying its hot-file cutoff while grouping
(`ignore_file_freq=4`, `min_chain_size=3`):

```bash
uv run python scripts/swebench/analyze_swebench_pro_chains.py \
  --ignore-file-freq 4 \
  --min-chain-size 3 \
  --compact-chain-ids \
  --output-path datasets/swebench_pro/cache/swe_bench_pro_issue_chains_deep.json \
  --nodes-output-path evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl
```

Use `--offline` when the commit-time cache is complete. Use
`--no-llm-noise-filter` for a model-free analysis with different semantics.
The analyzer loads the repository `.env` by default and follows the canonical
provider contract: `OPENAI_AUTH_TOKEN` for bearer auth and `OPENAI_BASE_URL` for
the endpoint. Override the dotenv path or provider flags only when intentionally
using another OpenAI-compatible service.
`instance_id` values must match the dataset split; a mismatch surfaces as
`missing_instance_ids` in the run's `experiment.json`.

Run the maintained SWE-bench unit-test set with:

```bash
uv run python -m unittest discover -s tests/unit -p 'test_swebench_*.py'
```
