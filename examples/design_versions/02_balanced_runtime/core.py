"""Compatibility re-export for the canonical balanced runtime.

The 02 design has been promoted into `src/simple_agent_lab/core.py`. Keeping
this tiny module lets older local demo imports (`from core import ...`) keep
working while the implementation has one source of truth.
"""

from simple_agent_lab.core import *  # noqa: F403
