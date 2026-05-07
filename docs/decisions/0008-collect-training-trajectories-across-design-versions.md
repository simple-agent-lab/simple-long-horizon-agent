# ADR 0008: Separate Trajectory, Evaluation, and Training Data Records

## Status

Accepted

## Context

ADR 0004 defines self-evolution as a harness loop:

```text
run -> trace -> evaluate -> propose candidate -> compare -> accept or reject
```

ADR 0005 makes `02_balanced_runtime` the lead core candidate for that work,
but `01_functional_loop` and `03_event_runtime` remain useful comparison
surfaces until the runtime sketches are unified.

The project needs three related but different artifacts:

- a trajectory that records what happened
- an evaluation result that scores a trajectory
- training examples derived from trajectory model turns plus optional eval labels

Keeping these separate matters because a trajectory should stay reusable even
when the scoring rule changes. A later training job should be able to attach a
new reward, preference label, or critique without rewriting the raw run record.

## Decision

Separate raw trajectories, evaluation results, and training examples.

The runtime-neutral records are:

- `trajectory`: raw run-level messages, events, and model turns for replay and debugging
- `eval_result`: score and metrics produced by a scorer over one trajectory
- `training_example`: one model-call input/output pair, optionally labeled by an eval result

The first local pipeline is:

```bash
PYTHONPATH=src python3 scripts/collect_design_version_trajectories.py
PYTHONPATH=src python3 evals/evaluate_design_version_traces.py
PYTHONPATH=src python3 scripts/export_training_examples.py
```

`runs/run_training_trace_eval.sh` runs the three steps together. Generated
JSONL under `evals/out/` is local artifact data, not source.

Runtime-specific behavior stays thin:

- `01_functional_loop`: collect model turns by wrapping the local `ModelFn`;
  the message list remains its trace.
- `02_balanced_runtime`: collect from `state.events`; this stays the lead
  self-evolution path for now.
- `03_event_runtime`: collect from its event log and `RunReport`; this stays
  the observability/provider-boundary reference for now.

Those design-version adapters are temporary producers, not the architecture.
Once the three runtime sketches collapse into one implementation, the adapters
should collapse too.

Do not introduce a long-lived trace database, benchmark registry, or
provider-specific training payload at this stage.

## Consequences

All three runtime designs can now produce comparable trajectories, eval
results, and training examples, but implementation effort still concentrates on
the balanced runtime until the runtime sketches are unified.

Future training work can start from JSONL records without scraping terminal
output. Later evals can add pairwise comparisons or richer metrics without
polluting the raw trajectory record.

The pipeline is intentionally deterministic and small. If later examples use
live models, they should add run metadata such as provider, model id, seed,
temperature, and prompt version before treating trajectories as training data.

The tradeoff is that the first eval is demo-specific. That is acceptable
because the goal is to establish the harness shape before building a benchmark
suite.

## Alternatives Considered

- Add separate eval frameworks inside each runtime. Rejected because it would
  split the comparison surface and encourage three independent frameworks.
- Only instrument `02_balanced_runtime`. Rejected because 01 and 03 are still
  useful baselines for runtime selection and teaching.
- Store provider wire payloads directly. Rejected because training data should
  start from provider-neutral project types and be converted at the provider or
  training boundary.
- Put score/reward fields directly on trajectories. Rejected because a
  trajectory should record what happened; eval and training labels are later
  interpretations.
- Emit only terminal traces. Rejected because terminal text is useful for
  humans but weak input for training and automated eval.
