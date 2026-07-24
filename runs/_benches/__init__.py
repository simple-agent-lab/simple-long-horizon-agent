"""Internal per-bench run modules — the public entry is ``runs/run_bench.py``.

Each module here exposes ``NAME`` / ``DESCRIPTION``, ``_build_parser()`` and
``run(args) -> dict``; ``runs/run_bench.py`` imports them to provide one CLI
over every bench. They keep a thin ``main()`` so they remain runnable for
debugging, but the supported entry point is ``run_bench.py <bench>``.
"""
