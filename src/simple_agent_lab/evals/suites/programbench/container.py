"""ProgramBench container half: the functions a suite supplies.

ProgramBench is a *reverse-engineering* benchmark (facebookresearch/programbench):
the workspace holds a compiled ``./executable`` plus its bundled docs, and the
agent must write a brand-new codebase from scratch whose ``./compile.sh``
rebuilds an executable with identical behavior — inferring that behavior only by
running ``./executable`` and reading the docs. Two facts make it differ from
SWE-bench and shape this module:

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
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.agent_flavors import flavor_from_env
from simple_agent_lab.agents.flavors import ArtifactPut, build_flavor_agent
from simple_agent_lab.core import Agent
from simple_agent_lab.hooks import HookContext, HookDecision, HookPoint
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import runtime_message
from simple_agent_lab.protocols import EventKind
from simple_agent_lab.tools import AgentTool, text_result

AGENT_NAME = "programbench_agent"
SUBMIT_TOOL_NAME = "submit"
AGENT_ROLE = (
    "Reverse-engineer ./executable in the workspace and write an original "
    "codebase plus a compile.sh that rebuilds an identical executable."
)

# The reverse-engineering rules, adapted from mini-swe-agent's ProgramBench
# system_template to the simple-agent-lab bash agent. The baseline's
# `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submit protocol is represented by
# our `submit` tool, and per-command network isolation replaces Docker
# `--network none` so model calls can still reach the provider.
AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer.\n\n"
    "This is a reverse-engineering benchmark. You are given a compiled binary "
    "and its documentation.\n"
    "Your job is to write a new, original codebase from scratch that produces "
    "an executable with identical behavior.\n"
    "You must figure out the behavior solely by running and interacting with "
    "the provided binary.\n"
    "Writing original code based on observed behavior is the entire point of "
    "this benchmark.\n\n"
    "Any approach that shortcuts this — finding existing source code, wrapping "
    "the binary,\n"
    "or installing the original tool — does not count as a solution. It will "
    "be detected by an\n"
    "automated judge, marked as a disqualification, and scored as a failure.\n"
    "The trace will be posted to an online wall of shame.\n\n"
    "<RULES>\n"
    "## What is not allowed\n\n"
    "### 1. Obtaining source code\n\n"
    "The only source of truth about what the executable does is the executable "
    "itself and its\n"
    "bundled documentation. You must not search the internet, package "
    "registries, or any external\n"
    "source for information about this project's source code. Even if you "
    "recognize what the\n"
    "executable is, you must reimplement it from behavioral observation alone."
    "\n\n"
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
    "`./executable` is for\n"
    "observation only — your final solution must not depend on it or any other "
    "pre-built version\n"
    "of the same tool at runtime.\n\n"
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
    "interacting with it\n"
    "through its normal user interface (CLI flags, stdin/stdout, etc.).\n"
    "- You MUST NOT decompile `./executable` or use disassemblers (objdump, "
    "Ghidra, etc.) on it\n"
    "- You MUST NOT use strace, ltrace, or similar tracing/instrumentation "
    "tools on `./executable`\n\n"
    "Note: this restriction applies ONLY to the provided `./executable`. You "
    "are free to use any\n"
    "analysis tools on binaries that you produce yourself during development."
    "\n\n"
    "## What IS allowed\n\n"
    "- Running the executable with any inputs, flags, and arguments to observe "
    "its behavior\n"
    "- Reading any documentation files bundled in the workspace\n"
    "</RULES>\n\n"
    "## Runtime notes\n\n"
    "When the workspace is ready to be graded, call the `submit` tool. Calling "
    "`submit` ends the run.\n"
    "Before submission, each response should take at least one concrete action "
    "with an available tool (`bash`, `read`, `task`, or `submit`). Do not end "
    "by sending a plain final text answer.\n"
    "The container stays online for model API calls, but each agent bash "
    "command runs in a network-isolated namespace. Do not try to download "
    "anything; it is both disallowed and impossible from bash.\n"
    "Do not voluntarily stop with known gaps. A final answer that lists "
    "`Known limitations`, says core behavior is infeasible, or declares major "
    "commands best-effort is not an acceptable benchmark completion signal. "
    "When you discover a limitation, convert it into the next concrete "
    "observation, implementation, or verification step and keep improving the "
    "submission."
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
            check=False,
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

    # env-ok: host->container isolation handoff
    value = os.environ.get(REQUIRE_ISOLATION_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "off", "")


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    trace_put: ArtifactPut | None = None,
) -> Agent:
    """Build the ProgramBench agent: a bash agent whose commands are net-isolated.

    Probes ``unshare --net`` once; when available, every agent command runs in a
    network-less namespace (the model call still uses the container network).
    When it is unavailable this fails closed (raises) unless the caller opted out
    via ``--no-network-isolation`` (see ``_isolation_required``), so a run never
    silently loses ProgramBench's anti-cheat.

    The simple/workflow split is owned by ``build_flavor_agent``: this hook just
    injects the suite-specific knobs (net-isolation prefix, submit tool, runtime
    reminders, and the trace sink) in one call and lets the flavor builder drop
    whichever ones it does not use (e.g. an arm flavor's facade owns the per-step
    breakdown + sub-traces, so it uses the trace sink and ignores the tools).
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
    flavor = flavor_from_env()
    exec_prefix = NET_ISOLATION_PREFIX if isolated else ()
    # Pass both the simple-only knobs (hooks/submit tool) and the workflow-only
    # knob (trace_put) in one call; build_flavor_agent owns the simple/workflow
    # split and drops whichever knobs the selected flavor does not use.
    return build_flavor_agent(
        flavor=flavor,
        name=AGENT_NAME,
        provider=provider,
        cwd=cwd,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        request_extra=request_extra,
        hooks=_runtime_reminder_hooks(context or {}),
        tools=(make_submit_tool(),),
        trace_put=trace_put,
        bash_exec_prefix=exec_prefix,
    )


def make_submit_tool() -> AgentTool:
    """Return the ProgramBench submit tool.

    mini-swe-agent submits ProgramBench runs by executing
    ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``. In this runtime a terminating
    tool is the equivalent explicit completion signal.
    """

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: Any,
        on_update: Any,
    ) -> Any:
        del call_id, abort, on_update
        summary = str(args.get("summary") or "").strip()
        if not summary:
            return text_result(
                "Missing required submit argument: summary.",
                is_error=True,
            )
        return text_result(
            "Submission accepted. The workspace will be packaged for grading.\n"
            f"Summary: {summary}",
            terminate=True,
        )

    return AgentTool(
        name=SUBMIT_TOOL_NAME,
        description=(
            "Submit the ProgramBench workspace for grading after compile.sh "
            "rebuilds ./executable and you have verified the behavior as far as "
            "possible. Calling this ends the run."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the implementation and verification.",
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def _runtime_reminder_hooks(
    context: Mapping[str, Any],
) -> dict[HookPoint, tuple[Any, ...]]:
    runtime = context.get("runtime")
    if not isinstance(runtime, Mapping):
        return {}
    hook = _runtime_reminder_hook(runtime)
    return {HookPoint.POST_TOOL_USE: (hook,)}


def _runtime_reminder_hook(runtime: Mapping[str, Any]) -> Any:
    def remind(ctx: HookContext) -> HookDecision | None:
        if ctx.tool_call is not None and ctx.tool_call.name == SUBMIT_TOOL_NAME:
            return None

        messages: list[str] = []
        step_warning = _step_limit_warning(ctx, runtime)
        if step_warning:
            messages.append(step_warning)
        time_warning = _wall_time_warning(runtime)
        if time_warning:
            messages.append(time_warning)
        if not messages:
            return None
        return HookDecision(
            emit_messages=(
                runtime_message(
                    "\n\n".join(messages),
                    sender="programbench",
                    target=ctx.agent,
                    kind="context",
                ),
            )
        )

    return remind


def _step_limit_warning(ctx: HookContext, runtime: Mapping[str, Any]) -> str:
    max_turns = _positive_int(runtime.get("max_turns"))
    if max_turns is None:
        return ""
    n_model_calls = sum(
        1 for event in ctx.state.events if event.kind == EventKind.MODEL_RESPONSE
    )
    remaining = max_turns - n_model_calls
    if remaining >= 20:
        return ""
    return (
        "<IMPORTANT>\n"
        "There is a limit to the steps you can take. You are now "
        f"{max(0, remaining)} steps away from reaching your limit. After you "
        "reach your limit your current solution will be auto-submitted.\n"
        "At this point, please abort any specific issues that you are debugging "
        "or solving and focus on the big picture. Please make sure that\n"
        "1. Your solution compiles and produces an executable (it's ok if it is "
        "still missing functionality)\n"
        "2. If there are any steps left to do, or limitations that you are aware "
        'of, please write them to a document "AGENT_REPORT.md". Focus on handing '
        "off to the next agent, i.e., focus on clearly describing the problems "
        "and any todo items that are left over.\n"
        "</IMPORTANT>"
    )


def _wall_time_warning(runtime: Mapping[str, Any]) -> str:
    wall_time_seconds = _positive_float(runtime.get("wall_time_seconds"))
    started = _positive_float(runtime.get("started_monotonic"))
    if wall_time_seconds is None or started is None:
        return ""
    remaining = wall_time_seconds - (time.monotonic() - started)
    if remaining >= 600:
        return ""
    minutes = max(0, int(remaining / 60))
    return (
        "<IMPORTANT>\n"
        "You are running low on time. You have approximately "
        f"{minutes} minutes remaining before timeout.\n"
        "Please wrap up your work now:\n"
        "1. Ensure your solution compiles and produces an executable (it's ok if "
        "it is still missing functionality)\n"
        "2. If there are any steps left to do, or limitations that you are aware "
        'of, please write them to a document "AGENT_REPORT.md". Focus on handing '
        "off to the next agent, i.e., focus on clearly describing the problems "
        "and any todo items that are left over.\n"
        "3. Submit with the `submit` tool.\n"
        "</IMPORTANT>"
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


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
    the recommended workflow, command discipline, and explicit submit step —
    adapted to this runtime's ``submit`` tool instead of mini-swe-agent's
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` command.
    """

    tui_hint = (
        "- You SHOULD extensively test the executable to understand its "
        "behavior before writing code.\n"
        "  If you are dealing with a TUI, tmux/libtmux has been installed to "
        "help you test/inspect/it."
        if shutil.which("tmux")
        else "- You SHOULD extensively test the executable to understand its "
        "behavior before writing code."
    )
    return "\n".join(
        [
            "## Task context",
            "",
            "We want to write the source code for a given executable.",
            "The executable is located at `./executable` in the workspace root "
            f"(`{workdir}`).",
            "",
            "You also have access to the existing documentation.",
            "",
            "## Your task",
            "",
            "Implement the source code to generate an executable of exactly "
            "identical behavior as the original.",
            "",
            "No project-specific dependencies are pre-installed.",
            "You do NOT have access to the internet from bash commands.",
            "**IMPORTANT**: Make sure that the executable(s) and everything else "
            "that is an artifact is not committed, i.e., is in your `.gitignore` "
            "file.",
            "Finally, commit your changes.",
            "",
            "Make sure that you have a `./compile.sh` file that produces an "
            "executable `./executable` in the workspace root.",
            "`compile.sh` should be executable and should install any "
            "dependencies needed to compile the executable.",
            "If your compile.sh fails to compile on a fresh checkout, your task "
            "has failed.",
            "",
            "## Important: This is a reverse-engineering benchmark",
            "",
            "Your goal is to write original code from scratch that reproduces "
            "the executable's behavior.",
            "The only way to learn what the executable does is to run it and "
            "read its bundled documentation.",
            "",
            "Any attempt to obtain source code — whether successful or not — or "
            "to wrap/reuse the",
            "provided binary will be detected by an automated judge, "
            "disqualified, and scored as zero.",
            "See the full rules in the system prompt above. Key points:",
            "",
            "- Do NOT search the internet, clone repos, or download the project "
            "from any package registry",
            "- Do NOT wrap, shim, or delegate to the provided `./executable` or "
            "any installed version of the same tool",
            "- Do NOT decompile the provided `./executable` or use strace/ltrace "
            "on it (analyzing your own binaries is fine)",
            tui_hint,
            "",
            "## Recommended Workflow",
            "",
            "1. Explore all documentation files",
            "2. Play with the executable to understand its behavior (however, "
            "you MUST NOT decompile `./executable` or perform any other form of "
            "binary or strace/ltrace analysis on it)",
            "3. Write the source code to implement the behavior",
            "4. If you find a missing feature, parse error, flag, output mode, "
            "or command behavior, do not summarize it as a limitation and stop; "
            "implement the closest faithful behavior you can infer, add a "
            "focused check, and repeat.",
            "",
            "## Command Execution Rules",
            "",
            "You can execute bash commands and edit files to implement the "
            "necessary changes.",
            "",
            "You are operating in an environment where",
            "",
            "1. You issue tool calls",
            "2. The system executes the command(s) or tool action(s)",
            "3. You see the result(s)",
            "4. You write your next command(s)",
            "",
            "Each response before submission should include:",
            "",
            "1. **Reasoning text** where you explain your analysis and plan",
            "2. At least one tool call with your command or action",
            "",
            "**CRITICAL REQUIREMENTS:**",
            "",
            "- Your response SHOULD include reasoning text explaining what "
            "you're doing",
            "- Your response MUST include AT LEAST ONE tool call before "
            "submission (`bash`, `read`, `task`, or `submit`)",
            "- Directory or environment variable changes are not persistent. "
            "Every bash action is executed in a new subshell.",
            "- However, you can prefix any bash action with "
            "`MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or "
            "write/load environment variables from files",
            "- Submit your changes and finish your work by calling the `submit` "
            "tool. Do not combine submission with any other action. "
            "<important>After this tool call, you cannot continue working on "
            "this task.</important>",
            "",
            "Example of a CORRECT response:",
            "<example_response>",
            "I need to understand the structure of the repository first. Let "
            "me check what files are in the current directory to get a better "
            "understanding of the codebase.",
            "",
            '[Makes bash tool call with {"command": "ls -la"} as arguments]',
            "</example_response>",
            "",
            "<system_information>",
            _system_info(),
            "</system_information>",
            "",
            "## Useful command examples",
            "",
            "python is available as python3",
            "",
            "### Create a new file:",
            "",
            "```bash",
            "cat <<'EOF' > newfile.py",
            "import numpy as np",
            'hello = "world"',
            "print(hello)",
            "EOF",
            "```",
            "",
            "### Edit files with sed:",
            "",
            "```bash",
            "# Replace all occurrences",
            "sed -i 's/old_string/new_string/g' filename.py",
            "",
            "# Replace only first occurrence",
            "sed -i 's/old_string/new_string/' filename.py",
            "",
            "# Replace first occurrence on line 1",
            "sed -i '1s/old_string/new_string/' filename.py",
            "",
            "# Replace all occurrences in lines 1-10",
            "sed -i '1,10s/old_string/new_string/g' filename.py",
            "```",
            "",
            "### View file content:",
            "",
            "```bash",
            "# View specific lines with numbers",
            "nl -ba filename.py | sed -n '10,20p'",
            "```",
            "",
            "### Any other command you want to run",
            "",
            "```bash",
            "anything",
            "```",
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
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "--allow-empty", "-m", "baseline")
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
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
