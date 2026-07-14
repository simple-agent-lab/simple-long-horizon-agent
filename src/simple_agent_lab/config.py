"""Central registry for environment-backed configuration.

Every env knob the package reads is declared here, once, as an `EnvVar`: its
name, default, parser, docs, and a `group` placing it in the config hierarchy.
One resolver (`EnvVar.get`) applies one precedence rule — an explicit override
beats the environment, which beats the declared default — so callers stop
re-deriving `os.environ.get(...) + coerce + default` in a dozen places, and a
profile or CLI flag becomes just an override layer fed to the same resolver.
See ADR centralized-env-config.

`group` is a dotted ``domain.subsystem`` label (the config hierarchy); the
top-level domains are ``agent``, ``eval``, ``provider``, ``trace``,
``runtime``. Classification is by **domain**, not by an env var's name prefix:
the workflow-arm knobs were renamed from ``SWE_*`` to ``SAL_WORKFLOW_*`` so the
name no longer implies they are SWE-bench-specific — they configure the agent's
workflow arm (``agent.workflow``). ``SWE_REPO_LANGUAGE`` keeps its ``SWE_``
prefix because it really is eval-container config (``eval.swebench``).

This module is a FOUNDATION-zone leaf: it imports nothing internal, so every
layer can read config through it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_UNSET = object()


def as_int(*, minimum: int | None = None) -> Callable[[str], int]:
    """A parser reading an int, clamped to ``minimum`` when given."""

    def parse(raw: str) -> int:
        value = int(raw)
        return value if minimum is None else max(minimum, value)

    return parse


def as_float(*, minimum: float | None = None) -> Callable[[str], float]:
    """A parser reading a float, clamped to ``minimum`` when given."""

    def parse(raw: str) -> float:
        value = float(raw)
        return value if minimum is None else max(minimum, value)

    return parse


def as_bool(raw: str) -> bool:
    """Truthy reading matching the env knobs (``1``/``true``/``yes``/``on``)."""

    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EnvVar:
    """One environment-backed config value, declared once.

    ``group`` is a dotted ``domain.subsystem`` label (the config hierarchy);
    ``parse`` turns the raw string into the typed value. ``get`` applies the
    one precedence rule: an explicit ``default`` argument wins, else the
    environment, else the declared default. A blank or unparseable value falls
    back to the default and never raises — matching the ad-hoc readers this
    replaces.
    """

    name: str
    default: Any
    group: str
    doc: str
    parse: Callable[[str], Any] = str

    def get(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        default: Any = _UNSET,
    ) -> Any:
        if default is not _UNSET:
            return default
        source = os.environ if environ is None else environ
        raw = (source.get(self.name) or "").strip()
        if not raw:
            return self.default
        try:
            return self.parse(raw)
        except (ValueError, TypeError):
            return self.default


# --------------------------------------------------------------------------- #
# agent.workflow — knobs for the loop / PDR workflow arms, read in the agent
# build layer. Generic to any suite that runs these arms; renamed from the
# misleading ``SWE_*`` prefix (see ADR centralized-env-config).
# --------------------------------------------------------------------------- #
WORKER_MAX_TURNS = EnvVar(
    "SAL_WORKFLOW_WORKER_MAX_TURNS",
    40,
    "agent.workflow",
    "Per-worker inner turn budget.",
    as_int(minimum=1),
)
LOOP_MAX_TURNS = EnvVar(
    "SAL_WORKFLOW_LOOP_MAX_TURNS",
    6,
    "agent.workflow",
    "Loop workflow: judge-gated outer iterations.",
    as_int(minimum=1),
)
PDR_ROUNDS = EnvVar(
    "SAL_WORKFLOW_PDR_ROUNDS",
    2,
    "agent.workflow",
    "PDR workflow: distill/refine rounds.",
    as_int(minimum=1),
)
PDR_WIDTH = EnvVar(
    "SAL_WORKFLOW_PDR_WIDTH",
    3,
    "agent.workflow",
    "PDR workflow: parallel attempts per round.",
    as_int(minimum=1),
)
PDR_ATTEMPT_TURNS = EnvVar(
    "SAL_WORKFLOW_PDR_ATTEMPT_TURNS",
    None,
    "agent.workflow",
    "PDR workflow: per-attempt turn budget; defaults to SAL_WORKFLOW_WORKER_MAX_TURNS.",
    as_int(minimum=1),
)

# --------------------------------------------------------------------------- #
# agent.compression — context-compression knobs for the default agent.
# --------------------------------------------------------------------------- #
COMPRESSION_THRESHOLD = EnvVar(
    "SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS",
    None,
    "agent.compression",
    "Token threshold that triggers compression; default is window * ratio, "
    "else a fixed fallback.",
    as_int(minimum=0),
)
COMPRESSION_WINDOW_RATIO = EnvVar(
    "SAL_AGENT_COMPRESSION_WINDOW_RATIO",
    0.8,
    "agent.compression",
    "Fraction of the context window used as the threshold when none is set.",
    as_float(),
)
COMPRESSION_KEEP_RECENT = EnvVar(
    "SAL_AGENT_COMPRESSION_KEEP_RECENT",
    4,
    "agent.compression",
    "Recent turns kept verbatim during compression.",
    as_int(minimum=0),
)

# --------------------------------------------------------------------------- #
# agent.llm — request-level knobs for LLM-backed agents.
# --------------------------------------------------------------------------- #
LLM_REQUEST_TIMEOUT = EnvVar(
    "SAL_LLM_REQUEST_TIMEOUT_SECONDS",
    None,
    "agent.llm",
    "Per-request model API timeout in seconds; unset uses the adapter default.",
    as_float(minimum=1.0),
)

# --------------------------------------------------------------------------- #
# agent.tools — shared tool knobs.
# --------------------------------------------------------------------------- #
BASH_DEFAULT_TIMEOUT = EnvVar(
    "SAL_BASH_DEFAULT_TIMEOUT_SECONDS",
    None,
    "agent.tools",
    "Default timeout for bash tool commands; unset uses the tool default.",
    as_float(minimum=1.0),
)
BASH_MAX_TIMEOUT = EnvVar(
    "SAL_BASH_MAX_TIMEOUT_SECONDS",
    None,
    "agent.tools",
    "Maximum model-selectable timeout for bash tool commands; unset uses the tool default.",
    as_float(minimum=1.0),
)
BASH_MAX_OUTPUT_CHARS = EnvVar(
    "SAL_BASH_MAX_OUTPUT_CHARS",
    None,
    "agent.tools",
    "Maximum model-visible characters per bash output stream; unset uses the tool default.",
    as_int(minimum=1),
)
BASH_SUBMISSION_MARKER = EnvVar(
    "SAL_BASH_SUBMISSION_MARKER",
    "",
    "agent.tools",
    "When set, a bash command whose first output line matches this marker terminates the run and stores the remaining output as a submission.",
)

# --------------------------------------------------------------------------- #
# eval.swebench — SWE-bench container knobs.
# --------------------------------------------------------------------------- #
REPO_LANGUAGE = EnvVar(
    "SWE_REPO_LANGUAGE",
    "python",
    "eval.swebench",
    "Repo language hint for the SWE-bench container.",
)

# --------------------------------------------------------------------------- #
# eval.onemillion — OneMillion workflow-flavor tuning knobs. (The workflow is
# selected with AGENT_FLAVOR, like every suite; these only tune the chosen one.)
# --------------------------------------------------------------------------- #
OMB_REFLECTION_ROUNDS = EnvVar(
    "OMB_REFLECTION_ROUNDS",
    2,
    "eval.onemillion",
    "Reflection rounds.",
    as_int(minimum=1),
)
OMB_PARALLEL_WORKERS = EnvVar(
    "OMB_PARALLEL_WORKERS", 3, "eval.onemillion", "Parallel workers.", as_int(minimum=1)
)
OMB_PDR_ROUNDS = EnvVar(
    "OMB_PDR_ROUNDS", 2, "eval.onemillion", "PDR rounds.", as_int(minimum=1)
)
OMB_PDR_WIDTH = EnvVar(
    "OMB_PDR_WIDTH", 3, "eval.onemillion", "PDR width.", as_int(minimum=1)
)
OMB_TIMEOUT = EnvVar(
    "OMB_TIMEOUT",
    600.0,
    "eval.onemillion",
    "Per-request timeout for every sub-agent (seconds).",
    as_float(minimum=1.0),
)

# --------------------------------------------------------------------------- #
# trace — trajectory tracing.
# --------------------------------------------------------------------------- #
LIVE_TRACE_PATH = EnvVar(
    "LIVE_TRACE_PATH",
    None,
    "trace",
    "Bind-mounted path for incremental live trace output (unset = off).",
)

# Every declared var, so the catalog (docs/agent-native/configuration.md) can be
# generated from / validated against this list. New EnvVars must be added here.
REGISTRY: tuple[EnvVar, ...] = (
    WORKER_MAX_TURNS,
    LOOP_MAX_TURNS,
    PDR_ROUNDS,
    PDR_WIDTH,
    PDR_ATTEMPT_TURNS,
    COMPRESSION_THRESHOLD,
    COMPRESSION_WINDOW_RATIO,
    COMPRESSION_KEEP_RECENT,
    LLM_REQUEST_TIMEOUT,
    BASH_DEFAULT_TIMEOUT,
    BASH_MAX_TIMEOUT,
    BASH_MAX_OUTPUT_CHARS,
    BASH_SUBMISSION_MARKER,
    REPO_LANGUAGE,
    OMB_REFLECTION_ROUNDS,
    OMB_PARALLEL_WORKERS,
    OMB_PDR_ROUNDS,
    OMB_PDR_WIDTH,
    OMB_TIMEOUT,
    LIVE_TRACE_PATH,
)
