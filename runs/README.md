# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_examples.sh
bash runs/run_design_versions.sh
```

There is no default test runner yet. Testing and feedback are documented as a
first-priority design constraint, and the concrete suite will be added after the
core architecture is settled.

This runs the tiny message-runtime demo:

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
