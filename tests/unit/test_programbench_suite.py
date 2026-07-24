"""Unit-smoke for the ProgramBench adapter (ADR 0022). No Docker, no programbench.

Four seams, none of which need the optional ``programbench`` package or Docker:

- the host half is data — ``ProgrambenchSuite.launch_spec`` resolves the image,
  workdir, ``network_mode=host`` and ``cap_add=("SYS_ADMIN",)`` as a value, and
  ``task_input`` drops gold/identity fields (an instance carrying ``image_name``
  lets ``launch_spec`` skip importing ``programbench.constants``);
- the container half's pure functions — ``build_task`` states the rules,
  ``extract_result`` packs the whole workspace into a decodable tarball, and
  ``prepare`` inits a repo with a repo-local identity (never ``--global``);
- the per-command network isolation wiring — ``build_agent`` records whether
  ``unshare --net`` is available (failing closed when it is missing unless the
  caller opts out), and the bash ``exec_prefix`` it relies on really wraps the
  launched argv; and
- one end-to-end in-process run via ``LocalProcessBackend`` + the fake provider,
  proving ``result.json`` carries a decodable submission tarball of the workspace.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from simple_agent_lab.agent_flavors import AGENT_FLAVOR_ENV
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.core import Agent
from simple_agent_lab.evals import (
    RESULT_KEY,
    LocalDirStore,
    LocalProcessBackend,
    Suite,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.programbench import container
from simple_agent_lab.hooks import HookContext, HookPoint
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import ToolCallBlock
from simple_agent_lab.protocols import ModelResponseEvent
from simple_agent_lab.state import State

from evals.programbench import harness
from evals.programbench.evaluate_submissions import default_eval_dir
from evals.programbench.suite import ProgrambenchSuite

PROGRAMBENCH_CONTAINER = "simple_agent_lab.evals.suites.programbench.container"
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def _instance(**overrides: Any) -> dict[str, Any]:
    """A ProgramBench-shaped record.

    Carries ``image_name`` so ``launch_spec`` resolves the image without
    importing the (optional) ``programbench`` package, plus the gold/identity
    fields the loader injects so the sanitization path has something to drop.
    """

    base: dict[str, Any] = {
        "instance_id": "testorg__calculator.abc1234",
        "image_name": "programbench/testorg_1776_calculator.abc1234",
        "language": "bash",
        "difficulty": "easy",
        # gold / project-identity fields the agent must not see:
        "repository": "testorg/calculator",
        "commit": "abc1234567890",
        "branches": {"main": {}},
        "tests": {"main": [{"name": "t"}]},
    }
    base.update(overrides)
    return base


class ProgrambenchSuiteDriverTest(unittest.TestCase):
    def test_launch_spec_is_data(self) -> None:
        suite = ProgrambenchSuite()
        self.assertIsInstance(suite, Suite)
        self.assertEqual(suite.name, "programbench")
        self.assertEqual(suite.container_module, PROGRAMBENCH_CONTAINER)

        spec = suite.launch_spec(_instance())
        self.assertEqual(spec.workdir, "/workspace")
        self.assertEqual(spec.shell, ("bash", "-lc"))
        self.assertEqual(
            spec.image, "programbench/testorg_1776_calculator.abc1234:task_cleanroom"
        )
        # Host network keeps the model API reachable during the run; SYS_ADMIN is
        # what powers the per-command `unshare --net` isolation in the container.
        self.assertEqual(spec.network_mode, "host")
        self.assertEqual(spec.cap_add, ("SYS_ADMIN",))
        self.assertEqual(spec.nano_cpus, 20_000_000_000)
        self.assertEqual(spec.mem_limit, "60g")
        self.assertEqual(spec.memswap_limit, "60g")

    def test_task_input_drops_gold_and_identity_fields(self) -> None:
        view = ProgrambenchSuite().task_input(_instance())
        self.assertEqual(view["instance_id"], "testorg__calculator.abc1234")
        self.assertIn("language", view)  # toolchain hint is allowed, not gold
        for hidden in ("repository", "commit", "image_name", "branches", "tests"):
            self.assertNotIn(hidden, view)

    def test_eval_inputs_is_none(self) -> None:
        # ProgramBench scores via the official CLI on the host, not in-environment,
        # so no gold is staged and the container half exposes no `evaluate` hook.
        self.assertIsNone(ProgrambenchSuite().eval_inputs(_instance()))

    def test_no_network_isolation_drops_the_cap(self) -> None:
        # The `--no-network-isolation` run path withholds CAP_SYS_ADMIN.
        spec = ProgrambenchSuite(cap_add=()).launch_spec(_instance())
        self.assertEqual(spec.cap_add, ())

    def test_prepare_node_runtime_verifies_and_extracts_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / harness.NODE_DIST_NAME / "bin"
            source.mkdir(parents=True)
            node = source / "node"
            node.write_bytes(b"fake-linux-node")
            archive = root / "fixture.tar.xz"
            with tarfile.open(archive, mode="w:xz") as bundle:
                bundle.add(
                    source.parent,
                    arcname=harness.NODE_DIST_NAME,
                )
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            binary_checksum = hashlib.sha256(node.read_bytes()).hexdigest()
            wheelhouse = root / "wheelhouse"

            def download(_url: str, destination: Path) -> None:
                shutil.copyfile(archive, destination)

            installed = harness.prepare_node_runtime(
                wheelhouse,
                url="https://example.invalid/node.tar.xz",
                sha256=checksum,
                binary_sha256=binary_checksum,
                downloader=download,
            )
            self.assertEqual(installed.name, "node")
            self.assertEqual(installed.read_bytes(), b"fake-linux-node")

            installed.write_bytes(b"poisoned-cache")
            restored = harness.prepare_node_runtime(
                wheelhouse,
                url="https://example.invalid/node.tar.xz",
                sha256=checksum,
                binary_sha256=binary_checksum,
                downloader=download,
            )
            self.assertEqual(restored.read_bytes(), b"fake-linux-node")

            victim = root / "victim"
            victim.write_bytes(b"must-not-execute")
            restored.unlink()
            restored.symlink_to(victim)
            replaced = harness.prepare_node_runtime(
                wheelhouse,
                url="https://example.invalid/node.tar.xz",
                sha256=checksum,
                binary_sha256=binary_checksum,
                downloader=download,
            )
            self.assertFalse(replaced.is_symlink())
            self.assertEqual(replaced.read_bytes(), b"fake-linux-node")
            self.assertEqual(victim.read_bytes(), b"must-not-execute")

            bad_wheelhouse = root / "bad-wheelhouse"
            bad_wheelhouse.mkdir()
            escaped = root / "escaped"
            escaped.mkdir()
            (bad_wheelhouse / harness.NODE_DIST_NAME).symlink_to(
                escaped, target_is_directory=True
            )
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                harness.prepare_node_runtime(
                    bad_wheelhouse,
                    url="https://example.invalid/node.tar.xz",
                    sha256=checksum,
                    binary_sha256=binary_checksum,
                    downloader=download,
                )
            self.assertFalse((escaped / "bin").exists())


class ProgrambenchEvaluationScriptTest(unittest.TestCase):
    def test_default_eval_dir_isolates_single_instance_reruns(self) -> None:
        run_root = Path("evals/out/programbench")

        self.assertEqual(
            default_eval_dir(run_root, "run-1", None),
            run_root / "run-1_eval",
        )
        self.assertEqual(
            default_eval_dir(run_root, "run-1", ["canop__broot.d6c798e"]),
            run_root / "run-1_eval_canop__broot.d6c798e",
        )


class ProgrambenchContainerHalfTest(unittest.TestCase):
    def test_system_prompt_rejects_known_limitations_as_completion(self) -> None:
        prompt = container.AGENT_SYSTEM_PROMPT.lower()
        self.assertIn("known limitations", prompt)
        self.assertIn("not an acceptable benchmark completion signal", prompt)
        self.assertIn("keep improving", prompt)
        self.assertIn("submit", prompt)

    def test_submit_tool_terminates_the_run(self) -> None:
        tool = container.make_submit_tool()
        self.assertEqual(tool.name, container.SUBMIT_TOOL_NAME)

        missing = tool.execute("call-1", {}, lambda: False, None)
        self.assertTrue(missing.is_error)
        self.assertFalse(missing.terminate)

        result = tool.execute(
            "call-2",
            {"summary": "built and checked"},
            lambda: False,
            None,
        )
        self.assertFalse(result.is_error)
        self.assertTrue(result.terminate)

    def test_runtime_reminder_hook_emits_low_step_warning(self) -> None:
        hooks = container._runtime_reminder_hooks(
            {"runtime": {"max_turns": 5, "wall_time_seconds": None}}
        )
        hook = hooks[HookPoint.POST_TOOL_USE][0]
        state = State(task="task")
        state.record_event(
            ModelResponseEvent(
                agent="programbench_agent",
                output_kind="step",
                target="programbench_agent",
                tool_call_count=1,
            )
        )

        decision = hook(
            HookContext(
                point=HookPoint.POST_TOOL_USE,
                agent="programbench_agent",
                state=state,
                tool_call=ToolCallBlock("call-1", "bash", {}),
            )
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        text = decision.emit_messages[0].content[0].text
        self.assertIn("steps away", text)
        self.assertIn("AGENT_REPORT.md", text)

    def test_build_agent_installs_runtime_reminder_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                container, "_detect_network_isolation", return_value=True
            ):
                agent = container.build_agent(
                    provider=FAKE_PROVIDER,
                    cwd=Path(tmp),
                    context={"runtime": {"max_turns": 5}},
                )

        self.assertIn(HookPoint.POST_TOOL_USE, agent.hooks)

    def test_build_task_states_the_rules(self) -> None:
        task = container.build_task(_instance(), workdir="/workspace")
        self.assertIsInstance(task, str)
        low = task.lower()
        self.assertIn("compile.sh", low)
        self.assertIn("/workspace", task)
        self.assertIn("reverse-engineer", low)
        # The task tells the agent its commands have no internet.
        self.assertIn("access to the internet", low.replace("-", " "))
        # Known gaps should drive more work, not become an early final answer.
        self.assertIn("limitation", low)
        self.assertIn("do not summarize it", low)
        self.assertIn("implement", low)
        self.assertIn("submit", low)

    def test_build_task_reports_real_container_system_info(self) -> None:
        # build_task runs in-container, so it states the true OS/arch via
        # os.uname() — more accurate than mini-swe-agent's host-rendered {{system}}.
        task = container.build_task(_instance(), workdir="/workspace")
        self.assertIn("<system_information>", task)
        self.assertIn(os.uname().sysname, task)

    def test_build_task_mentions_tmux_only_when_available(self) -> None:
        # The TUI hint is probed, not assumed: present iff `tmux` is on PATH.
        with mock.patch.object(container.shutil, "which", return_value="/usr/bin/tmux"):
            with_tmux = container.build_task(_instance(), workdir="/workspace")
        with mock.patch.object(container.shutil, "which", return_value=None):
            without_tmux = container.build_task(_instance(), workdir="/workspace")
        self.assertIn("tmux", with_tmux.lower())
        self.assertNotIn("tmux", without_tmux.lower())

    def test_extract_result_packs_the_whole_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "compile.sh").write_text("#!/bin/sh\ngcc -o executable main.c\n")
            (ws / "main.c").write_text("int main(void){return 0;}\n")
            result = container.extract_result(ws, _instance())

        self.assertEqual(result["instance_id"], "testorg__calculator.abc1234")
        self.assertGreater(result["submission_tar_bytes"], 0)
        raw = base64.b64decode(result["submission_tar_b64"])
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            names = set(tar.getnames())
        # arcname="." reproduces the official `tar -czf -C <ws> .` layout.
        self.assertIn("./compile.sh", names)
        self.assertIn("./main.c", names)

    def test_prepare_inits_repo_with_local_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            context = container.prepare(ws, _instance())
            self.assertEqual(context, {})
            self.assertTrue((ws / ".git").is_dir())
            # The identity is repo-local, written into .git/config — never global.
            local_config = (ws / ".git" / "config").read_text(encoding="utf-8")
            self.assertIn("simple-agent-lab", local_config)
            head = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=ws,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(head.returncode, 0)


class NetworkIsolationWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        # build_agent flips a module global; reset it so order can't leak between
        # tests.
        container._network_isolation_active = None
        self.addCleanup(setattr, container, "_network_isolation_active", None)

    def test_build_agent_records_isolation_when_unshare_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                container, "_detect_network_isolation", return_value=True
            ):
                agent = container.build_agent(provider=FAKE_PROVIDER, cwd=Path(tmp))
            self.assertIsNotNone(agent)
            self.assertTrue(container._network_isolation_active)
            # extract_result echoes the recorded status into the result.
            result = container.extract_result(Path(tmp), _instance())
        self.assertTrue(result["network_isolated"])

    def test_build_agent_uses_unified_simple_flavors_and_default_compression(
        self,
    ) -> None:
        provider = Provider(
            id="fake",
            api="fake",
            model="fake-model",
            context_window=200_000,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(os.environ, {AGENT_FLAVOR_ENV: "bash_task_read"}),
                mock.patch.object(
                    container, "_detect_network_isolation", return_value=True
                ),
            ):
                agent = container.build_agent(provider=provider, cwd=Path(tmp))

        self.assertIsInstance(agent, Agent)
        self.assertIn(container.SUBMIT_TOOL_NAME, {tool.name for tool in agent.tools})
        self.assertIsNotNone(agent.context_policy)
        strategy = agent.context_policy.strategy
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 160_000)

    def test_build_agent_uses_unified_workflow_flavors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(os.environ, {AGENT_FLAVOR_ENV: "loop"}),
                mock.patch.object(
                    container, "_detect_network_isolation", return_value=True
                ),
            ):
                agent = container.build_agent(provider=FAKE_PROVIDER, cwd=Path(tmp))

        self.assertIsInstance(agent, Agent)

    def test_build_agent_falls_back_when_isolation_opted_out(self) -> None:
        # Explicit opt-out (REQUIRE_ISOLATION_ENV false-y, set by
        # --no-network-isolation): a missing `unshare --net` degrades to
        # un-isolated commands instead of aborting.
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(os.environ, {container.REQUIRE_ISOLATION_ENV: "0"}),
                mock.patch.object(
                    container, "_detect_network_isolation", return_value=False
                ),
            ):
                container.build_agent(provider=FAKE_PROVIDER, cwd=Path(tmp))
            self.assertFalse(container._network_isolation_active)
            result = container.extract_result(Path(tmp), _instance())
        self.assertFalse(result["network_isolated"])

    def test_container_environment_passes_agent_flavor(self) -> None:
        env = {
            "OPENAI_MODEL": "m",
            "OPENAI_AUTH_TOKEN": "tok",
            AGENT_FLAVOR_ENV: "loop",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            passed = harness.container_environment("openai")
        self.assertEqual(passed[AGENT_FLAVOR_ENV], "loop")

    def test_container_environment_passes_compression_knobs(self) -> None:
        env = {
            "OPENAI_MODEL": "m",
            "OPENAI_AUTH_TOKEN": "tok",
            "SAL_AGENT_COMPRESSION_WINDOW_RATIO": "0.02",
            "SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS": "20000",
            "SAL_AGENT_COMPRESSION_KEEP_RECENT": "2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            passed = harness.container_environment("openai")
        self.assertEqual(passed["SAL_AGENT_COMPRESSION_WINDOW_RATIO"], "0.02")
        self.assertEqual(passed["SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS"], "20000")
        self.assertEqual(passed["SAL_AGENT_COMPRESSION_KEEP_RECENT"], "2")

    def test_build_agent_fails_closed_by_default(self) -> None:
        # No opt-out (variable unset): a missing `unshare --net` aborts the run
        # rather than silently dropping ProgramBench's anti-cheat.
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(os.environ, clear=False),
                mock.patch.object(
                    container, "_detect_network_isolation", return_value=False
                ),
            ):
                os.environ.pop(container.REQUIRE_ISOLATION_ENV, None)
                with self.assertRaises(RuntimeError):
                    container.build_agent(provider=FAKE_PROVIDER, cwd=Path(tmp))
            # The probe result is recorded before the guard fires.
            self.assertFalse(container._network_isolation_active)

    def test_exec_prefix_actually_wraps_the_bash_argv(self) -> None:
        """The isolation mechanism is the bash tool's ``exec_prefix``. Prove the
        prefix is really prepended to the launched argv (so ``unshare --net``
        would wrap the command) using a network-free observable wrapper: ``env``
        injecting a variable the inner ``bash -lc`` echoes back."""

        from simple_agent_lab.tools.bash import run_bash

        with tempfile.TemporaryDirectory() as tmp:
            wrapped = run_bash(
                "echo netns=$SAL_NETNS",
                cwd=tmp,
                exec_prefix=("env", "SAL_NETNS=isolated"),
            )
            plain = run_bash("echo netns=$SAL_NETNS", cwd=tmp)

        self.assertIn("netns=isolated", wrapped.stdout)
        self.assertNotIn("isolated", plain.stdout)
        # The recorded (model-visible) command is the raw string, prefix-free.
        self.assertEqual(wrapped.command, "echo netns=$SAL_NETNS")

    def test_net_isolation_prefix_raises_lo_then_execs_command(self) -> None:
        """The prefix seals capabilities in a new user+network namespace and
        then executes the command through the loopback setup shim."""

        from simple_agent_lab.tools.bash import run_bash

        prefix = container.NET_ISOLATION_PREFIX
        self.assertEqual(
            prefix[:5],
            (
                "/usr/bin/setpriv",
                "--reuid=65534",
                "--regid=65534",
                "--clear-groups",
                "/usr/bin/unshare",
            ),
        )
        self.assertIn("--map-root-user", prefix)
        self.assertIn("--net", prefix)
        wrapper = prefix[prefix.index("/bin/sh") :]
        self.assertIn('exec "$@"', " ".join(wrapper))

        with tempfile.TemporaryDirectory() as tmp:
            out = run_bash("echo wrapped-ok", cwd=tmp, exec_prefix=wrapper)
        self.assertIn("wrapped-ok", out.stdout)


class ProgrambenchEndToEndTest(unittest.TestCase):
    """Real host + container half in-process (fake provider, no Docker).

    The network probe is forced off; in-process (no container) can't isolate, so
    the run opts out explicitly (``REQUIRE_ISOLATION_ENV=0``) — otherwise
    ``build_agent`` fails closed by default — and never depends on
    ``unshare --net`` being permitted in the test environment.
    """

    def test_in_process_run_packs_a_decodable_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            (ws / "executable").write_text("#!/bin/sh\necho hi\n")
            (ws / "README.md").write_text("bundled docs\n")
            root = Path(tmp) / "runs"
            store = LocalDirStore(root)

            with (
                mock.patch.dict(os.environ, {container.REQUIRE_ISOLATION_ENV: "0"}),
                mock.patch.object(
                    container, "_detect_network_isolation", return_value=False
                ),
            ):
                artifacts = run_suite_instance(
                    suite=ProgrambenchSuite(),
                    instance=_instance(),
                    backend=LocalProcessBackend(workspace=ws),
                    store=store,
                    run_root=root,
                    run_id="pb",
                    provider="fake",
                    max_turns=3,
                )

            self.assertEqual(artifacts.status_code, 0)
            bound = store.bind(artifacts.run_dir)
            result = json.loads(bound.get(RESULT_KEY).decode("utf-8"))
            self.assertEqual(result["instance_id"], "testorg__calculator.abc1234")
            self.assertFalse(result["network_isolated"])

            raw = base64.b64decode(result["submission_tar_b64"])
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("./executable", names)
            self.assertIn("./README.md", names)

            # The agent-visible instance had its gold/identity fields stripped.
            agent_view = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertNotIn("repository", agent_view)
            self.assertNotIn("commit", agent_view)
