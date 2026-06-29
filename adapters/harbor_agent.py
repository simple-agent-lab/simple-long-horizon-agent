"""Harbor adapter: run Simple Agent Lab on Harbor/SWE-Marathon benchmarks.

This is a Harbor "installed agent" (see harbor.agents.installed.base). Harbor
installs Simple Agent Lab into the task container and invokes the agent loop
in-process there. The host-side class only orchestrates install + invoke; all
agent logic runs inside the container, so the bash tool naturally operates on
the task workspace.

Usage:
    harbor run -p tasks/find-network-alignments \
        --agent-import-path adapters.harbor_agent:SimpleAgentLab \
        --model openai/gpt-5.5 \
        --ae OPENAI_AUTH_TOKEN=$OPENAI_AUTH_TOKEN

Model: pass `--model openai/<model_id>`. The provider prefix is consumed by
Harbor's model parsing; <model_id> is forwarded to the container as
OPENAI_MODEL. simple-agent-lab reads the OpenAI token from OPENAI_AUTH_TOKEN
(not OPENAI_API_KEY); this adapter forwards the host's OPENAI_API_KEY (or
OPENAI_AUTH_TOKEN) into the container under that name.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


# Where to install simple-agent-lab from in the container. Override with the
# SIMPLE_AGENT_LAB_SOURCE env var to point at a fork/branch/local wheel, e.g.
# "git+https://github.com/simple-agent-lab/simple-agent-lab.git@main".
DEFAULT_SOURCE = "git+https://github.com/simple-agent-lab/simple-agent-lab.git"

# Cap on agent turns; the task's agent.timeout_sec is the real backstop.
MAX_TURNS = 2000

# SWE-Marathon tasks use several workspace roots. The in-container script
# probes these after honoring SAL_WORKDIR.
WORKDIR_CANDIDATES = (
    "/workspace/rust-java-lsp",
    "/app/rj-rust",
    "/workspace",
    "/app",
)

# Long-horizon SWE-Marathon tasks benefit from the task sub-agent by default.
DEFAULT_AGENT_FLAVOR = "bash_task"

ATIF_PATH = Path("trajectory.json")


class SimpleAgentLab(BaseInstalledAgent):
    """Run simple-agent-lab's bash agent inside a Harbor task container."""

    @staticmethod
    @override
    def name() -> str:
        return "simple-agent-lab"

    @override
    def version(self) -> str | None:
        return "0.1.0"

    @override
    def get_version_command(self) -> str | None:
        return 'python3 -c \'import simple_agent_lab; print(getattr(simple_agent_lab, "__version__", "0"))\''

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "if command -v python3 >/dev/null 2>&1 "
                "&& python3 -m pip --version >/dev/null 2>&1; then "
                "true; else "
                "apt-get update && apt-get install -y --no-install-recommends "
                "python3-pip; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        source = self._get_env("SIMPLE_AGENT_LAB_SOURCE") or DEFAULT_SOURCE
        pip_args = self._get_env("SIMPLE_AGENT_LAB_PIP_ARGS") or ""
        await self.exec_as_agent(
            environment,
            command=(
                "python3 - <<'PY'\n"
                "import shlex\n"
                "import subprocess\n"
                "import sys\n"
                f"source = {source!r}\n"
                f"extra_args = {pip_args!r}\n"
                "cmd = [sys.executable, '-m', 'pip', 'install', '--user', "
                "'--break-system-packages', *shlex.split(extra_args), source]\n"
                "subprocess.check_call(cmd)\n"
                "PY"
            ),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self._parsed_model_name:
            raise ValueError(
                "Model name must be in the format openai/<model_id>; "
                f"got {self.model_name!r}"
            )

        env = self._build_container_env()
        # Write the instruction to a file to avoid shell-quoting a 6h agent
        # loop seed string.
        await self.exec_as_agent(
            environment,
            command=f"printf %s {shlex.quote(instruction)} > /tmp/sal_instruction.txt",
        )
        await self.exec_as_agent(
            environment,
            command=self._container_command(),
            env=env,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory_path = self.logs_dir / ATIF_PATH
        if not trajectory_path.exists():
            return
        try:
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        metrics = payload.get("final_metrics")
        if not isinstance(metrics, dict):
            return
        context.n_input_tokens = _optional_int(metrics.get("total_prompt_tokens"))
        context.n_cache_tokens = _optional_int(metrics.get("total_cached_tokens"))
        context.n_output_tokens = _optional_int(metrics.get("total_completion_tokens"))
        context.cost_usd = _optional_float(metrics.get("total_cost_usd"))
        extra = payload.get("extra")
        if isinstance(extra, dict):
            metadata = dict(context.metadata or {})
            for key in ("sal_trace_path", "sal_raw_trace_path", "workdir"):
                if key in extra:
                    metadata[key] = extra[key]
            context.metadata = metadata or context.metadata

    def _build_container_env(self) -> dict[str, str]:
        host_token = self._get_env("OPENAI_AUTH_TOKEN") or self._get_env(
            "OPENAI_API_KEY"
        )
        flavor = (
            self._get_env("SAL_AGENT_FLAVOR")
            or self._get_env("AGENT_FLAVOR")
            or DEFAULT_AGENT_FLAVOR
        )
        max_turns = self._get_env("SAL_MAX_TURNS") or str(MAX_TURNS)
        env: dict[str, str] = {
            "OPENAI_MODEL": self._parsed_model_name,
            "SAL_AGENT_FLAVOR": flavor,
            "AGENT_FLAVOR": flavor,
            "SAL_MAX_TURNS": max_turns,
        }
        if host_token:
            env["OPENAI_AUTH_TOKEN"] = host_token
        for key in (
            "OPENAI_BASE_URL",
            "API_KIND",
            "OPENAI_SESSION_ID",
            "OPENAI_LOG_ID",
            "OPENAI_REASONING_EFFORT",
            "SAL_WORKDIR",
            "SIMPLE_AGENT_LAB_PRICE_BOOK",
            "SIMPLE_AGENT_LAB_CONTEXT_WINDOW_BOOK",
            "FAKE_PROVIDER",
        ):
            value = self._get_env(key)
            if value:
                env[key] = value
        reasoning = self._get_env("REASONING_EFFORT") or self._get_env(
            "OPENAI_REASONING_EFFORT"
        )
        if reasoning:
            env["REASONING_EFFORT"] = reasoning
        return env

    def _container_script(self) -> str:
        candidates = ", ".join(repr(path) for path in WORKDIR_CANDIDATES)
        return f"""python3 - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

from simple_agent_lab.agent_flavors import SIMPLE_AGENT_FLAVORS
from simple_agent_lab.agents.flavors import build_flavor_agent
from simple_agent_lab.llm.env import provider_from_env
from simple_agent_lab.trace import (
    atif_trajectory_from_run,
    event_stream,
    run_trace_from_state,
    write_jsonl,
)

LOG_DIR = Path('/logs/agent')
SAL_TRACE_PATH = Path('/logs/agent/sal/trajectory.jsonl')
SAL_RAW_TRACE_PATH = Path('/logs/agent/sal/trajectory.jsonl.raw.jsonl')
ATIF_TRACE_PATH = Path('/logs/agent/trajectory.json')
WORKDIR_CANDIDATES = ({candidates},)


def resolve_workdir():
    explicit = os.environ.get('SAL_WORKDIR')
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise SystemExit(f'SAL_WORKDIR does not exist or is not a directory: {{explicit}}')
        return path

    cwd = Path.cwd()
    candidates = []
    if str(cwd) not in ('/', '/root', '/installed-agent'):
        candidates.append(cwd)
    candidates.extend(Path(path) for path in WORKDIR_CANDIDATES)
    for path in candidates:
        if path.is_dir():
            return path
    checked = ', '.join(str(path) for path in candidates)
    raise SystemExit(f'Could not detect task workspace; set SAL_WORKDIR. Checked: {{checked}}')


def package_version():
    try:
        return importlib.metadata.version('simple-agent-lab')
    except importlib.metadata.PackageNotFoundError:
        return '0'


def write_outputs(state, workdir, flavor):
    SAL_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    trace_id = (
        os.environ.get('SAL_TRACE_ID')
        or os.environ.get('OPENAI_LOG_ID')
        or os.environ.get('OPENAI_SESSION_ID')
        or f'harbor-sal-{{uuid.uuid4().hex}}'
    )
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer='harbor:SimpleAgentLab',
        meta={{
            'agent_flavor': flavor,
            'model': os.environ.get('OPENAI_MODEL', ''),
            'workdir': str(workdir),
        }},
    )
    header, lines, raw_pool = event_stream(trace)
    write_jsonl(SAL_TRACE_PATH, [header, *lines])
    if raw_pool:
        write_jsonl(SAL_RAW_TRACE_PATH, raw_pool)

    atif = atif_trajectory_from_run(
        trace_id=trace_id,
        task=trace.task,
        events=trace.events,
        messages=trace.messages,
        agent_name='simple-agent-lab',
        agent_version=package_version(),
        model_name=os.environ.get('OPENAI_MODEL', ''),
        producer='harbor:SimpleAgentLab',
        extra={{
            'agent_flavor': flavor,
            'workdir': str(workdir),
            'sal_trace_path': 'sal/trajectory.jsonl',
            'sal_raw_trace_path': 'sal/trajectory.jsonl.raw.jsonl' if raw_pool else None,
        }},
    )
    ATIF_TRACE_PATH.write_text(
        json.dumps(atif, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


provider = provider_from_env()
workdir = resolve_workdir()
os.chdir(workdir)
flavor = (os.environ.get('SAL_AGENT_FLAVOR') or os.environ.get('AGENT_FLAVOR') or {DEFAULT_AGENT_FLAVOR!r}).strip().lower()
if flavor not in SIMPLE_AGENT_FLAVORS:
    raise SystemExit(f'Unsupported SAL_AGENT_FLAVOR={{flavor!r}}; expected one of {{SIMPLE_AGENT_FLAVORS}}.')
agent = build_flavor_agent(
    flavor=flavor,
    provider=provider,
    cwd=workdir,
    name='simple_agent_lab',
)
with open('/tmp/sal_instruction.txt', encoding='utf-8') as f:
    task = f.read()
state, events = agent.run(
    task,
    max_turns=int(os.environ.get('SAL_MAX_TURNS', {MAX_TURNS!r})),
)
for _event in events:
    pass
write_outputs(state, workdir, flavor)
PY"""

    def _container_command(self) -> str:
        return f"(\n{self._container_script()}\n) 2>&1 | tee /logs/agent/sal.log"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
