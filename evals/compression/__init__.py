"""Compression eval suite.

Behavior checks for the two context-compression strategies that ship in
`simple_agent_lab.compression`:

- `ToolCompactStrategy` (rule-based, no LLM)
- `SummarizeStrategy` (LLM-backed)

The suite answers the questions the unit tests deliberately skip. Unit
tests pin *mechanics* with `threshold_tokens=1` and fake compressors;
this suite measures *effectiveness*: is the token estimate that drives
`threshold_tokens` close to real provider tokens, how much does each
strategy actually shrink a realistic transcript, are tool pairs and
pinned kinds preserved, and does the summary prompt retain the durable
facts it promises to keep.

`scenarios` and `metrics` are model-free so they can run in CI. The live
half (real provider token counts, real summary fidelity) lives in
`run_eval` behind a `--live` flag and the OpenAI env vars.
"""
