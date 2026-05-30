# Compression Eval Output

Local artifacts from the compression eval suite (`evals/compression/`).
Contents are gitignored except this README.

## Directory Layout

```text
evals/out/compression/
├── README.md                 ← this file
└── <run-id>/
    └── eval_results.jsonl     ← one `eval_result` record per check
```

`<run-id>` defaults to a `YYYYMMDD-HHMMSS` timestamp; override with
`--run-id`.

## Record Shape

Each line is a project-owned `eval_result` (schema
`simple-agent-lab.evaluation.v1`), the same shape the SWE-bench adapter
writes:

```json
{
  "schema": "simple-agent-lab.evaluation.v1",
  "type": "eval_result",
  "trace_id": "planning-dialogue/summary-fidelity",
  "scorer": "summary_fact_recall",
  "passed": true,
  "score": 1.0,
  "metrics": { "semantic_recall": 1.0, "literal_recall": 1.0, "ratio": 0.25 },
  "reason": "semantic recall 100% (literal 100%) of 5 facts; ratio 0.25",
  "meta": { "summary_text": "..." }
}
```

## Reproducing

```bash
# Offline only (no model, runs in CI):
bash runs/run_compression_eval.sh

# Offline + live (needs OPENAI_MODEL / OPENAI_AUTH_TOKEN):
bash runs/run_compression_eval.sh --live
```
