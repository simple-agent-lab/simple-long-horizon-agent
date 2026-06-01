"""ClawEvalkit integration for Simple Agent Lab.

Provides an adapter layer that lets Simple Agent Lab's Bash Agent evaluate
ClawEvalkit benchmarks. The adapter converts between SAL's runtime events
and ClawEvalkit's transcript format, handles scoring, and manages task lifecycle.

Usage:
    python -m evals.openclaw --bench pinchbench --model claude-sonnet --sample 3
"""

from .runner import run_eval

__all__ = ["run_eval"]
