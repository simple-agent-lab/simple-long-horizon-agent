# GDPVal Output

Local GDPVal artifacts are ignored by git. A run uses the standard suite layout:

```text
evals/out/gdpval/
  instance_<id>.jsonl        # optional cached input row
  <run-id>/
    <task-id>/
      input/instance.json
      out/result.json
      out/trajectory.jsonl
      out/workspace.tar.gz   # Docker/local-store runs
    judge_summary.jsonl      # when runs/run_gdpval.py --judge is used
    judge_summary.json       # aggregate judge summary
  <run-id>-judge/
    <task-id>/
      input/instance.json
      out/result.json        # rubric score for the candidate
      out/trajectory.jsonl
```

The solver `result.json` contains produced file metadata and workspace archive
information. When `--judge` is used, the judge run's `result.json` contains the
rubric score, and the solver run directory receives aggregate judge summaries.
