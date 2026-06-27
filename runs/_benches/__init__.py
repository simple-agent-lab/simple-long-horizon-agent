"""Internal per-bench run modules — the public entry is ``runs/run_bench.py``.

Each module here exposes ``NAME`` / ``DESCRIPTION``, ``_build_parser()`` and
``run(args) -> dict``; ``runs/run_bench.py`` imports them to provide one CLI
over every bench. They keep a thin ``main()`` so they remain runnable for
debugging, but the supported entry point is ``run_bench.py <bench>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable (``evals`` at the root, ``simple_agent_lab`` under
# ``src``) regardless of how a submodule is loaded.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
