"""Container halves of shipped suites.

These modules run *inside* the eval container, so they ship in the wheel and
import only the standard library plus the installed ``simple-long-horizon-agent``
runtime. A suite's host half (image/launch/prediction shape) lives next to its
adapter under ``evals/<suite>/``; its container half lives here so the
in-container runner imports it with zero file copying.
"""
