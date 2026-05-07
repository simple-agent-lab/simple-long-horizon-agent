# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_examples.sh
bash runs/run_design_versions.sh
bash runs/run_self_evolution_probe.sh
bash runs/run_training_trace_eval.sh
```

The first focused tests cover the promoted balanced runtime:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

This runs the recipe demo on the canonical runtime:

```bash
python3 scripts/run_tiny_demo.py --recipe all
```

To inspect context management behavior:

```bash
python3 scripts/run_tiny_demo.py --recipe debate --last-messages 1
```

To compare the three architecture sketches:

```bash
bash runs/run_design_versions.sh
```

To run the first self-evolution harness probe:

```bash
bash runs/run_self_evolution_probe.sh
```

To run the full local harness pipeline for the three design-version demos:

```bash
bash runs/run_training_trace_eval.sh
```

That script performs three separate steps:

```bash
python3 scripts/collect_design_version_trajectories.py
python3 evals/evaluate_design_version_traces.py
python3 scripts/export_training_examples.py
```
