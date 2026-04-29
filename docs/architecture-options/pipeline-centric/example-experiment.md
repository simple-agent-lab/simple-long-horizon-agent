# Example Experiment: Research Map-Reduce

This experiment tests whether parallel specialized research improves final answer quality.

## Goal

Answer one research question using multiple independent research agents and a synthesizer.

## Pipeline

```text
question
  -> researcher_a
  -> researcher_b
  -> researcher_c
  -> synthesizer
  -> verifier
  -> final_answer
```

## Nodes

```text
researcher_a
  Focus: definitions and background.

researcher_b
  Focus: evidence and examples.

researcher_c
  Focus: limitations and counterarguments.

synthesizer
  Focus: combine notes into a coherent answer.

verifier
  Focus: flag unsupported claims and missing caveats.
```

## Artifacts

- `research_note`
- `synthesis_draft`
- `verification_report`
- `final_answer`

## Evaluation

Compare against a single-agent baseline on:

- Coverage.
- Coherence.
- Unsupported claims.
- Cost.
- Latency.

