"""ProgramBench container half: the functions a suite supplies.

ProgramBench is a *reverse-engineering* benchmark (facebookresearch/programbench):
the workspace holds a compiled ``./executable`` plus its bundled docs, and the
agent must write a brand-new codebase from scratch whose ``./compile.sh``
rebuilds an executable with identical behavior — inferring that behavior only by
running ``./executable`` and reading the docs. Two facts make it differ from
SWE-bench and shape this module (see
``programbench-reverse-engineering-adapter``):

- The run's **product is the whole workspace**, not a ``git diff``. The container
  half can only hand bytes back through ``out/result.json``, so ``extract_result``
  tars + gzips the workspace and returns it base64-encoded under
  ``submission_tar_b64``; the host decodes it into the ``<id>/submission.tar.gz``
  layout the official ProgramBench evaluator expects.
- ProgramBench's anti-cheat relies on the agent having **no network** while it
  works. Our agent runs *inside* the container and must reach the model API, so
  instead of ``--network none`` we keep the container online but run **every
  agent bash command in a network-isolated namespace** (``unshare --net``), built
  in ``build_agent``. Model calls keep the network; agent commands do not.

It imports only the standard library and the installed wheel (``core``, ``llm``,
``llm_agent``, ``tools.bash``), so it runs inside any ProgramBench image with no
copied files. Scoring is the official ProgramBench evaluator on the host
(``evals/programbench/evaluate_submissions.py``), so there is no ``evaluate``
hook here and the host stages no ``eval_inputs``.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools.bash import make_bash_tool

AGENT_NAME = "programbench_agent"
AGENT_ROLE = (
    "Reverse-engineer ./executable in the workspace and write an original "
    "codebase plus a compile.sh that rebuilds an identical executable."
)

# The reverse-engineering rules, adapted from ProgramBench's system template to
# the simple-agent-lab bash agent: dropped mini-swe-agent-only mechanics (the
# COMPLETE_TASK submit command, observation/step-limit templating) and added the
# fact that the bash tool has no network here.
AGENT_SYSTEM_PROMPT = (
    "You are a software engineer working inside a ProgramBench "
    "reverse-engineering container through the bash tool.\n\n"
    "The workspace contains a compiled program at `./executable` plus its "
    "bundled documentation. Your job is to write a brand-new, original codebase "
    "from scratch whose build produces an executable with identical behavior. "
    "You must infer that behavior ONLY by running and observing `./executable` "
    "and reading the bundled docs. Writing original code from observed behavior "
    "is the entire point of this benchmark.\n\n"
    "Any approach that shortcuts this — finding existing source code, wrapping "
    "the binary, or installing the original tool — does not count as a "
    "solution. It is detected by an automated judge, marked as a "
    "disqualification, and scored as a failure. "
    "The trace will be posted to an online wall of shame.\n\n"
    "Each bash call runs in a fresh shell rooted at the workspace, so include "
    "any cd or env setup in the command and use non-interactive flags (`-y`, "
    "`--no-pager`; avoid `vi`/`nano`). Independent read-only commands may run in "
    "parallel; never run parallel writes against the same file.\n\n"
    "The bash tool has NO network access — commands run in a network-isolated "
    "namespace — so do not try to download anything; it is both disallowed and "
    "impossible. Your model reasoning is separate and unaffected.\n\n"
    "## What is not allowed\n\n"
    "### 1. Obtaining source code\n\n"
    "The only source of truth about what the executable does is the executable "
    "itself and its bundled documentation. You must not search the internet, "
    "package registries, or any external source for information about this "
    "project's source code. Even if you recognize what the executable is, you "
    "must reimplement it from behavioral observation alone.\n\n"
    "This includes but is not limited to:\n"
    "- Cloning or browsing the original GitHub repository, its forks, or mirrors\n"
    "- Downloading the project from package registries: `cargo install <project>`, "
    "`go get github.com/<org>/<project>`, `pip install <project>`, "
    "`apt-get source <project>`, `npm install <project>`, etc.\n"
    "- Fetching source tarballs from project websites (e.g., "
    "`curl https://lua.org/ftp/lua-5.5.0.tar.gz`)\n"
    "- Using a package manager to download the project as a dependency and then "
    "reading its cached source (e.g., navigating into `~/.cargo/registry/src/` "
    "or `$(go env GOPATH)/pkg/mod/`)\n"
    "- Searching the web for the project's source code or implementation "
    "details\n\n"
    "### 2. Wrapping or reusing the original binary\n\n"
    "Your submission must be a genuine reimplementation. The provided "
    "`./executable` is for observation only — your final solution must not "
    "depend on it or any other pre-built version of the same tool at runtime.\n\n"
    "This includes but is not limited to:\n"
    "- Writing a wrapper script that delegates to the original binary "
    '(e.g., `exec zstd "$@"`)\n'
    "- Installing the tool from a package manager and shimming to it "
    "(e.g., `apt-get install nnn && cp $(which nnn) ./executable`)\n"
    "- Writing a `compile.sh` that simply makes the provided binary executable "
    "(`chmod +x ./executable`) or copies it (`cp ./executable ./executable`)\n"
    "- Building a binary whose main function shells out to an external tool "
    '(e.g., `Command::new("miniserve").args(args).exec()`)\n'
    "- Re-linking prebuilt `.o` object files found in the workspace without "
    "writing new source code\n\n"
    "### 3. Binary analysis of the provided executable\n\n"
    "All information about the provided `./executable` must be obtained by "
    "interacting with it through its normal user interface (CLI flags, "
    "stdin/stdout, etc.).\n"
    "- You MUST NOT decompile `./executable` or use disassemblers (objdump, "
    "Ghidra, etc.) on it\n"
    "- You MUST NOT use strace, ltrace, or similar tracing/instrumentation "
    "tools on `./executable`\n\n"
    "Note: this restriction applies ONLY to the provided `./executable`. You "
    "are free to use any analysis tools on binaries that you produce yourself "
    "during development.\n\n"
    "## What IS allowed\n\n"
    "- Running the executable with any inputs, flags, and arguments to observe "
    "its behavior\n"
    "- Reading any documentation files bundled in the workspace"
)

# Wrapper that runs each agent bash command in a fresh, network-less namespace.
# `unshare --net` needs CAP_SYS_ADMIN (the suite's launch_spec adds it). A brand
# new net namespace ships only a *down* loopback, so unlike `docker run --network
# none` (which auto-ups lo) `127.0.0.1` would be unusable inside it — breaking any
# command that binds localhost or starts a local server to self-test. We therefore
# slip a tiny `sh -c 'ip link set lo up; exec "$@"'` between `unshare` and the
# bootstrap-appended `bash -lc <cmd>`: it raises loopback first (best-effort —
# needs CAP_NET_ADMIN, which the container root has; silenced and `;`-chained so a
# failure never blocks the command), then execs the real command. `--` ends
# unshare's options; `_` is the `$0` placeholder so `"$@"` starts at `bash`.
NET_ISOLATION_PREFIX: tuple[str, ...] = (
    "unshare",
    "--net",
    "--",
    "sh",
    "-c",
    'ip link set lo up 2>/dev/null; exec "$@"',
    "_",
)

# Env var (default-closed gate): a failed `unshare --net` probe hard-fails the
# run unless an explicit opt-out (`--no-network-isolation`) sets this false-y.
REQUIRE_ISOLATION_ENV = "PROGRAMBENCH_REQUIRE_NET_ISOLATION"

# Set by build_agent so extract_result can record whether isolation was active.
_network_isolation_active: bool | None = None


def _detect_network_isolation() -> bool:
    """Probe whether ``unshare --net`` works in this container.

    Returns True when a no-network namespace can be created (CAP_SYS_ADMIN plus
    a permissive-enough kernel/seccomp), so agent commands can be isolated. When
    it is unavailable we fall back to plain bash and record it, rather than
    failing the run on an environment that cannot isolate.
    """

    try:
        proc = subprocess.run(
            ["unshare", "--net", "true"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _isolation_required() -> bool:
    """Whether a failed isolation probe should hard-fail the run (default True).

    ProgramBench's anti-cheat depends on agent commands having no network, so we
    fail closed: a missing ``unshare --net`` aborts the run unless the caller
    opts out explicitly (the run scripts set ``REQUIRE_ISOLATION_ENV`` to a
    false-y value when ``--no-network-isolation`` is passed). An unset variable
    counts as required, so silently un-isolated runs cannot happen by default.
    """

    value = os.environ.get(REQUIRE_ISOLATION_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "off", "")


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the ProgramBench agent: a bash agent whose commands are net-isolated.

    Probes ``unshare --net`` once; when available, every agent command runs in a
    network-less namespace (the model call still uses the container network).
    When it is unavailable this fails closed (raises) unless the caller opted out
    via ``--no-network-isolation`` (see ``_isolation_required``), so a run never
    silently loses ProgramBench's anti-cheat.
    """

    global _network_isolation_active
    isolated = _detect_network_isolation()
    _network_isolation_active = isolated
    if not isolated:
        if _isolation_required():
            raise RuntimeError(
                "ProgramBench requires per-command network isolation, but "
                "'unshare --net' is unavailable here. It needs CAP_SYS_ADMIN "
                "(the suite's launch_spec adds it) plus a kernel/daemon that "
                "permits new network namespaces. Running without it would give "
                "the agent's bash commands network access and weaken the "
                "anti-cheat. Fix the container capability, or pass "
                "--no-network-isolation to explicitly accept un-isolated commands."
            )
        print(
            "[programbench] WARNING: 'unshare --net' is unavailable; agent bash "
            "commands will run WITH network access (explicitly allowed via "
            "--no-network-isolation).",
            flush=True,
        )
    bash_tool = make_bash_tool(
        cwd=cwd,
        exec_prefix=NET_ISOLATION_PREFIX if isolated else (),
    )
    return make_llm_agent(
        name=AGENT_NAME,
        provider=provider,
        role=AGENT_ROLE,
        tools=[bash_tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
    )


def _system_info() -> str:
    """One-line OS/kernel/arch summary, read *inside* the container.

    mini-swe-agent renders ``{{system}}`` on the host (its model process), which
    can disagree with the container; build_task runs in-container, so ``os.uname()``
    reports the real environment the agent compiles for.
    """

    try:
        info = os.uname()
        return f"{info.sysname} {info.release} ({info.machine})"
    except (AttributeError, OSError):
        return "unknown"


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """Build the model-visible reverse-engineering task.

    ProgramBench's task is static (the per-instance signal is the workspace's
    ``./executable`` + docs, not any instance field), so this does not inject
    problem text the way SWE-bench injects ``problem_statement``.

    The content mirrors mini-swe-agent's ProgramBench ``instance_template`` —
    the recommended workflow and the "test the executable extensively before
    writing code" emphasis — but keeps our framework mechanics: the agent
    finishes by returning a summary, not mini-swe-agent's
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` command.
    """

    environment = [
        "## Environment",
        "- You are inside the ProgramBench container; the bash tool runs "
        f"locally in {workdir}.",
        f"- {workdir} contains a compiled `./executable` and its bundled "
        "documentation. There is NO original source code.",
        "- No project-specific dependencies are pre-installed, and the bash "
        "tool has no network access.",
        f"- System: {_system_info()}; `python3` is available.",
    ]
    if shutil.which("tmux"):
        environment.append(
            "- `tmux` is available — use it to drive and observe `./executable` "
            "(send keystrokes, capture panes) when it is an interactive/TUI "
            "program."
        )
    return "\n".join(
        [
            "Reverse-engineer the program in this ProgramBench instance.",
            "",
            *environment,
            "",
            "## Your task",
            "Implement original source code, from scratch, that produces an "
            "executable of exactly identical behavior as the provided "
            "`./executable`. The only way to learn what the executable does is "
            "to run it and read its bundled documentation.",
            "- Provide an executable `./compile.sh` at the workspace root that "
            "installs any needed build dependencies and produces `./executable` "
            "in the workspace root. If `compile.sh` fails on a fresh checkout, "
            "the task fails.",
            "- Keep build artifacts and the produced `./executable` out of git "
            "(add a `.gitignore`), then commit your source.",
            "",
            "## Important: this is a reverse-engineering benchmark",
            "Write original code from scratch that reproduces the executable's "
            "behavior. Any attempt to obtain source code — or to wrap, shim, or "
            "reuse the provided binary — is detected by an automated judge, "
            "disqualified, and scored as zero. The system message has the full "
            "rules; key points:",
            "- Do NOT search the internet, clone repos, or download the project "
            "from any package registry.",
            "- Do NOT wrap, shim, or delegate to the provided `./executable` or "
            "any installed version of the same tool.",
            "- Do NOT decompile the provided `./executable` or run strace/ltrace "
            "on it (analyzing your own binaries is fine).",
            "- You SHOULD extensively test the executable to understand its "
            "behavior before writing code.",
            "",
            "## Recommended workflow",
            "1. Explore all documentation files in the workspace.",
            "2. Play with `./executable` extensively — many inputs, flags, and "
            "stdin — to understand its behavior before writing any code.",
            "3. Write the source code to reproduce that behavior, then build it "
            "with `./compile.sh` and verify the output matches the original.",
            "",
            "## Final answer",
            "When `./compile.sh` rebuilds `./executable` and you have verified "
            "the behavior matches, return a short summary of what you built and "
            "how you checked it. Do NOT paste files — the harness packages your "
            "entire workspace automatically.",
        ]
    )


def prepare(workspace: Path, instance: Mapping[str, Any]) -> dict[str, Any]:
    """Init a repo + set a *repo-local* git identity so the agent's commits work.

    The product is the whole workspace, not git state, so this is best-effort:
    ProgramBench's official evaluator re-creates a synthetic repo if the
    submission lacks one. We still ``git init`` + set an identity so the prompt's
    "commit your source" step does not error mid-run. The identity is written
    repo-locally (never ``--global``, which would leak into the host ``~/.gitconfig``
    when this runs under the in-process backend), so the init must come first.
    """

    del instance
    workspace = Path(workspace)
    if not (workspace / ".git").exists():
        _git(workspace, "init")
    _git(workspace, "config", "user.email", "agent@simple-agent-lab.local")
    _git(workspace, "config", "user.name", "simple-agent-lab")
    return {}


def extract_result(
    workspace: Path,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack the whole workspace as the ProgramBench submission product.

    Mirrors the official runner's ``tar -czf`` of the workspace, but returns it
    base64-encoded in ``result.json`` (the only channel a container half has).
    The host decodes ``submission_tar_b64`` into ``<id>/submission.tar.gz`` for
    the official ProgramBench evaluator.
    """

    del context
    tar_b64, tar_bytes = _pack_workspace(Path(workspace))
    return {
        "instance_id": str(instance.get("instance_id") or ""),
        "submission_tar_b64": tar_b64,
        "submission_tar_bytes": tar_bytes,
        "network_isolated": _network_isolation_active,
    }


def _pack_workspace(workspace: Path) -> tuple[str, int]:
    """tar.gz the workspace into memory and base64-encode it.

    Uses stdlib ``tarfile`` (not the system ``tar``) so it works in any image.
    ``arcname="."`` reproduces the official ``tar -czf ... -C <workspace> .``
    layout the evaluator unpacks back into ``/workspace``.
    """

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(str(workspace), arcname=".")
    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def _git(workspace: Path, *args: str) -> None:
    """Best-effort git invocation; ProgramBench scoring never depends on it."""

    try:
        subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pass
