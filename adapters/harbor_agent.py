"""Harbor adapter: run simple-agent-lab's bash agent on Harbor benchmarks.

This is a Harbor "installed agent" (see harbor.agents.installed.base). Harbor
installs simple-agent-lab into the task container and invokes the bash agent
loop in-process there. The host-side class only orchestrates install + invoke;
all agent logic runs inside the container, so simple-agent-lab's bash tool
(executed via local subprocess) naturally operates on the container workspace.

Usage:
    harbor run -p tasks/rust-c-compiler \
        --agent adapters.harbor_agent:SimpleAgentLab \
        --model openai/gpt-5.5 \
        --allow-environment-host api.openai.com

Model: pass `--model openai/<model_id>`. The provider prefix is consumed by
Harbor's model parsing; <model_id> is forwarded to the container as
OPENAI_MODEL. simple-agent-lab reads the OpenAI token from OPENAI_AUTH_TOKEN
(not OPENAI_API_KEY); this adapter forwards the host's OPENAI_API_KEY (or
OPENAI_AUTH_TOKEN) into the container under that name.
"""

from __future__ import annotations

import os
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

# Workspace cwd inside the container (swe-marathon tasks mount under /app).
WORKDIR = "/app"


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
        return "python3 -c 'import simple_agent_lab; print(getattr(simple_agent_lab, \"__version__\", \"0\"))'"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y --no-install-recommends python3-pip",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        source = os.environ.get("SIMPLE_AGENT_LAB_SOURCE", DEFAULT_SOURCE)
        await self.exec_as_agent(
            environment,
            command=f"pip install --user --break-system-packages {shlex.quote(source)}",
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
        script = self._container_script()
        # Write the instruction to a file to avoid shell-quoting a 6h agent
        # loop seed string.
        await self.exec_as_agent(
            environment,
            command=f"printf %s {shlex.quote(instruction)} > /tmp/sal_instruction.txt",
        )
        await self.exec_as_agent(
            environment,
            command=f"{script} 2>&1 | tee /logs/agent/sal.log",
            env=env,
            cwd=WORKDIR,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        # No structured token/cost extraction yet; the tee'd log retains the
        # trajectory for offline analysis.
        pass

    def _build_container_env(self) -> dict[str, str]:
        host_token = os.environ.get("OPENAI_AUTH_TOKEN") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        env: dict[str, str] = {
            "OPENAI_MODEL": self._parsed_model_name,
            "OPENAI_AUTH_TOKEN": host_token,
        }
        if os.environ.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
        if os.environ.get("API_KIND"):
            env["API_KIND"] = os.environ["API_KIND"]
        return env

    def _container_script(self) -> str:
        return (
            "python3 - <<'PY'\n"
            "import os\n"
            "from simple_agent_lab.llm.env import provider_from_env\n"
            "from simple_agent_lab.agents.starter import make_bash_agent\n"
            "provider = provider_from_env()\n"
            f"agent = make_bash_agent(provider, cwd=os.environ.get('SAL_WORKDIR', {WORKDIR!r}))\n"
            "with open('/tmp/sal_instruction.txt') as f:\n"
            "    task = f.read()\n"
            f"state, events = agent.run(task, max_turns=int(os.environ.get('SAL_MAX_TURNS', {MAX_TURNS!r})))\n"
            "for _ in events:\n"
            "    pass\n"
            "PY"
        )
