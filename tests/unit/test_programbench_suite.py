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
  ``unshare --net`` is available, and the bash ``exec_prefix`` it relies on
  really wraps the launched argv; and
- one end-to-end in-process run via ``LocalProcessBackend`` + the fake provider,
  proving ``result.json`` carries a decodable submission tarball of the workspace.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from simple_agent_lab.evals import (
    RESULT_KEY,
    LocalDirStore,
    LocalProcessBackend,
    Suite,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.programbench import container
from simple_agent_lab.llm import Provider

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


class ProgrambenchContainerHalfTest(unittest.TestCase):
    def test_build_task_states_the_rules(self) -> None:
        task = container.build_task(_instance(), workdir="/workspace")
        self.assertIsInstance(task, str)
        low = task.lower()
        self.assertIn("compile.sh", low)
        self.assertIn("/workspace", task)
        self.assertIn("reverse-engineer", low)
        # The task tells the agent its commands have no network.
        self.assertIn("no network", low.replace("-", " "))

    def test_build_task_reports_real_container_system_info(self) -> None:
        # build_task runs in-container, so it states the true OS/arch via
        # os.uname() — more accurate than mini-swe-agent's host-rendered {{system}}.
        task = container.build_task(_instance(), workdir="/workspace")
        self.assertIn("System:", task)
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

    def test_build_agent_falls_back_without_unshare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                container, "_detect_network_isolation", return_value=False
            ):
                container.build_agent(provider=FAKE_PROVIDER, cwd=Path(tmp))
            self.assertFalse(container._network_isolation_active)
            result = container.extract_result(Path(tmp), _instance())
        self.assertFalse(result["network_isolated"])

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
        """NET_ISOLATION_PREFIX wraps `unshare --net` around a tiny sh that ups
        loopback then execs the command. `unshare --net` needs CAP_SYS_ADMIN we
        can't assume here, so check the structure, then prove the inner sh shim
        (everything after `unshare --net --`) still execs `bash -lc <cmd>` — the
        up-lo step fails harmlessly without a private netns thanks to `2>/dev/null`
        and `;`."""

        from simple_agent_lab.tools.bash import run_bash

        prefix = container.NET_ISOLATION_PREFIX
        self.assertEqual(prefix[:3], ("unshare", "--net", "--"))
        wrapper = prefix[3:]  # the `sh -c '...up lo...; exec "$@"' _` shim
        self.assertIn('exec "$@"', " ".join(wrapper))

        with tempfile.TemporaryDirectory() as tmp:
            out = run_bash("echo wrapped-ok", cwd=tmp, exec_prefix=wrapper)
        self.assertIn("wrapped-ok", out.stdout)


class ProgrambenchEndToEndTest(unittest.TestCase):
    """Real host + container half in-process (fake provider, no Docker).

    The network probe is forced off so the run never depends on ``unshare --net``
    being permitted in the test environment.
    """

    def test_in_process_run_packs_a_decodable_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            (ws / "executable").write_text("#!/bin/sh\necho hi\n")
            (ws / "README.md").write_text("bundled docs\n")
            root = Path(tmp) / "runs"
            store = LocalDirStore(root)

            with mock.patch.object(
                container, "_detect_network_isolation", return_value=False
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


if __name__ == "__main__":
    unittest.main()
