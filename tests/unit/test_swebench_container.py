"""SWE-bench container tests (deterministic; no network, no Docker).

The container is the single agent seam: simple flavors build one multi-turn
agent via the generic `agent_spec` path, while the workflow arms (loop / pdr)
build a facade through `build_agent`. These cover:

1. The workflow flavor worktree plumbing (`_add_worktree` / `_reset_worktree`)
   on a real temp repo — the genuinely error-prone mechanics that keep parallel
   rollouts from clobbering each other.
2. `build_agent` returning None for a simple flavor (so the framework falls
   through to the agent_spec path) and a facade for an arm.
3. The workflow flavor runners (`make_workflow_runner_for_flavor`) end to end,
   driving REAL bash agents with the deterministic fake adapter: a task carrying
   `<bash>...</bash>` makes the fake emit that command, which the bash tool runs
   in the agent's own cwd. That exercises the loop / pdr choreography (isolated
   edits, finalizer writing the workspace) without a model.

Also covers `run_pdr`'s worker-sequence form (the seam the arms rely on).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import Agent, Message, assistant_message
from simple_agent_lab.agent_flavors import (
    AGENT_FLAVOR_ENV,
    AGENT_FLAVORS,
    SIMPLE_AGENT_FLAVORS,
    WORKFLOW_AGENT_FLAVORS,
)
from simple_agent_lab.agents import flavors as af
from simple_agent_lab.agents.flavors import (
    build_flavor_agent,
    make_workflow_runner_for_flavor,
)
from simple_agent_lab.agents.starter import BASH_TASK_EXPLORER_ADDENDUM
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.evals.stores import container_store_from_env
from simple_agent_lab.evals.suites.swebench import container as wc
import simple_agent_lab.config as config
from simple_agent_lab.llm import Provider
from simple_agent_lab.state import State
from simple_agent_lab.workflow import run_pdr

FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")

# A task that makes the fake bash adapter run exactly this command (it extracts
# the text between <bash>...</bash>), creating `marker.txt` in the agent's cwd.
EDIT_TASK = "Solve this instance.\n<bash>echo edited > marker.txt</bash>"


def _git(args: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True, **kw
    )


def _init_repo() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="sal-wc-test-"))
    _git(["init", "-q"], workdir)
    _git(["config", "user.email", "t@t.invalid"], workdir)
    _git(["config", "user.name", "Test"], workdir)
    (workdir / "a.txt").write_text("original\n", encoding="utf-8")
    _git(["add", "-A"], workdir)
    _git(["commit", "-q", "-m", "baseline"], workdir)
    return workdir


# --------------------------------------------------------------------------- #
# Flavor selection: simple flavors delegate, arms get a facade
# --------------------------------------------------------------------------- #
class FlavorSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = _init_repo()

    def tearDown(self) -> None:
        subprocess.run(["rm", "-rf", str(self.workdir)], check=False)

    def test_simple_flavor_build_agent_returns_none(self) -> None:
        # Simple flavors are built by the generic agent_spec path; build_agent
        # declines them by returning None.
        for flavor in SIMPLE_AGENT_FLAVORS:
            with _envs({AGENT_FLAVOR_ENV: flavor}):
                self.assertIsNone(
                    wc.build_agent(provider=FAKE_PROVIDER, cwd=self.workdir),
                    flavor,
                )

    def test_arm_flavor_build_agent_returns_facade(self) -> None:
        for flavor in WORKFLOW_AGENT_FLAVORS:
            with _envs({AGENT_FLAVOR_ENV: flavor}):
                agent = wc.build_agent(provider=FAKE_PROVIDER, cwd=self.workdir)
                self.assertIsInstance(agent, Agent)

    def test_agent_spec_rejects_unknown_flavor(self) -> None:
        with _envs({AGENT_FLAVOR_ENV: "nonsense"}):
            with self.assertRaises(SystemExit):
                wc.agent_spec()

    def test_flavor_vocabulary_comes_from_top_level_module(self) -> None:
        # The runner's --agent-flavor choices (harness.AGENT_FLAVOR_CHOICES) must
        # match the shared vocabulary; this test keeps host and container choices
        # in sync without copying tuples between them.
        from evals.swebench.harness import AGENT_FLAVOR_CHOICES

        self.assertEqual(tuple(AGENT_FLAVOR_CHOICES), tuple(AGENT_FLAVORS))
        self.assertEqual(tuple(wc.ALL_FLAVORS), tuple(AGENT_FLAVORS))

    def test_agent_spec_keeps_capability_prompt_out_of_suite(self) -> None:
        with _envs({AGENT_FLAVOR_ENV: "bash_task_read"}):
            self.assertNotIn(
                BASH_TASK_EXPLORER_ADDENDUM,
                wc.agent_spec().system_prompt,
            )

    def test_agent_flavor_builder_adds_explorer_addendum_for_task_flavors(
        self,
    ) -> None:
        for flavor in ("bash_task", "bash_task_read"):
            agent = build_flavor_agent(
                flavor=flavor,
                provider=FAKE_PROVIDER,
                cwd=self.workdir,
                name="x",
                system_prompt="BASE",
            )
            self.assertIn(BASH_TASK_EXPLORER_ADDENDUM, agent.system_prompt)

    def test_bash_task_read_defaults_to_llm_compression(self) -> None:
        provider = Provider(
            id="fake",
            api="fake",
            model="fake-model",
            context_window=200_000,
        )
        agent = build_flavor_agent(
            flavor="bash_task_read",
            provider=provider,
            cwd=self.workdir,
            name="x",
            system_prompt="BASE",
        )

        self.assertIsNotNone(agent.context_policy)
        self.assertIsInstance(agent.context_policy.strategy, SummarizeStrategy)
        strategy = agent.context_policy.strategy
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 160_000)

    def test_default_compression_threshold_env_takes_precedence(self) -> None:
        provider = Provider(
            id="fake",
            api="fake",
            model="fake-model",
            context_window=200_000,
        )
        with _envs({config.COMPRESSION_THRESHOLD.name: "12345"}):
            agent = build_flavor_agent(
                flavor="bash_task_read",
                provider=provider,
                cwd=self.workdir,
                name="x",
                system_prompt="BASE",
            )

        self.assertIsNotNone(agent.context_policy)
        strategy = agent.context_policy.strategy
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 12_345)

    def test_default_compression_uses_litellm_window_book(self) -> None:
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="sal-window-book-")) / "windows.json"
        path.write_text(
            json.dumps(
                {
                    "my-model": {
                        "litellm_provider": "test",
                        "max_input_tokens": 300_000,
                    }
                }
            ),
            encoding="utf-8",
        )
        try:
            with _envs({"SIMPLE_AGENT_LAB_CONTEXT_WINDOW_BOOK": str(path)}):
                agent = build_flavor_agent(
                    flavor="bash_task_read",
                    provider=Provider(id="fake", api="fake", model="my-model"),
                    cwd=self.workdir,
                    name="x",
                    system_prompt="BASE",
                )
        finally:
            subprocess.run(["rm", "-rf", str(path.parent)], check=False)

        self.assertIsNotNone(agent.context_policy)
        strategy = agent.context_policy.strategy
        assert isinstance(strategy, SummarizeStrategy)
        self.assertEqual(strategy.threshold_tokens, 240_000)

    def test_default_compression_can_be_disabled(self) -> None:
        agent = build_flavor_agent(
            flavor="bash_task_read",
            provider=FAKE_PROVIDER,
            cwd=self.workdir,
            name="x",
            system_prompt="BASE",
            enable_default_compression=False,
        )

        self.assertIsNone(agent.context_policy)


# --------------------------------------------------------------------------- #
# Layer 1: git worktree plumbing
# --------------------------------------------------------------------------- #
class WorktreePlumbingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = _init_repo()
        self.baseline = af._baseline_commit(self.workdir)
        self.root = Path(tempfile.mkdtemp(prefix="sal-wc-wt-"))

    def tearDown(self) -> None:
        af._remove_worktrees(self.workdir, [], self.root)
        subprocess.run(["rm", "-rf", str(self.workdir)], check=False)

    def test_worktrees_isolate_edits_from_workspace_and_each_other(self) -> None:
        wt0 = af._add_worktree(self.workdir, self.baseline, self.root, 0)
        wt1 = af._add_worktree(self.workdir, self.baseline, self.root, 1)

        (wt0 / "new0.txt").write_text("zero\n", encoding="utf-8")
        (wt0 / "a.txt").write_text("changed-by-0\n", encoding="utf-8")
        (wt1 / "new1.txt").write_text("one\n", encoding="utf-8")

        # The canonical workspace is untouched while rollouts edit worktrees.
        self.assertFalse((self.workdir / "new0.txt").exists())
        self.assertFalse((self.workdir / "new1.txt").exists())
        self.assertEqual((self.workdir / "a.txt").read_text(), "original\n")
        # Worktrees don't see each other's files.
        self.assertFalse((wt0 / "new1.txt").exists())
        self.assertFalse((wt1 / "new0.txt").exists())

    def test_reset_returns_worktree_to_baseline(self) -> None:
        wt0 = af._add_worktree(self.workdir, self.baseline, self.root, 0)
        (wt0 / "scratch.txt").write_text("x\n", encoding="utf-8")
        (wt0 / "a.txt").write_text("dirty\n", encoding="utf-8")

        af._reset_worktree(wt0, self.baseline)

        self.assertFalse((wt0 / "scratch.txt").exists())
        self.assertEqual((wt0 / "a.txt").read_text(), "original\n")


# --------------------------------------------------------------------------- #
# Layer 2: arm runners end to end (real bash agents, fake adapter)
# --------------------------------------------------------------------------- #
class ArmRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = _init_repo()

    def tearDown(self) -> None:
        subprocess.run(["rm", "-rf", str(self.workdir)], check=False)

    def _no_leftover_worktrees(self) -> None:
        out = _git(["worktree", "list"], self.workdir).stdout
        # Only the main worktree should remain after cleanup.
        self.assertEqual(out.strip().count("\n"), 0, out)

    def test_pdr_finalizer_writes_workspace(self) -> None:
        with _envs(
            {
                config.PDR_ROUNDS.name: "1",
                config.PDR_WIDTH.name: "2",
                config.WORKER_MAX_TURNS.name: "3",
            }
        ):
            run = make_workflow_runner_for_flavor(
                "pdr",
                FAKE_PROVIDER,
                self.workdir,
                name=wc.AGENT_NAME,
                role=wc.AGENT_ROLE,
                system_prompt=wc.AGENT_SYSTEM_PROMPT,
                prepare_workspace=wc._prepare_workflow_workspace,
            )
            result = run(EDIT_TASK)
        # PDR's attempts ran in throwaway worktrees; the finalizer produced the
        # real edit in the canonical workspace.
        self.assertTrue((self.workdir / "marker.txt").exists())
        self.assertTrue(result.steps)
        self._no_leftover_worktrees()

    def test_loop_arm_runs_and_edits_workspace(self) -> None:
        with _envs(
            {config.LOOP_MAX_TURNS.name: "1", config.WORKER_MAX_TURNS.name: "2"}
        ):
            run = make_workflow_runner_for_flavor(
                "loop",
                FAKE_PROVIDER,
                self.workdir,
                name=wc.AGENT_NAME,
                role=wc.AGENT_ROLE,
                system_prompt=wc.AGENT_SYSTEM_PROMPT,
            )
            result = run(EDIT_TASK)
        self.assertTrue((self.workdir / "marker.txt").exists())
        self.assertTrue(result.steps)

    def test_unknown_arm_raises(self) -> None:
        with self.assertRaises(SystemExit):
            make_workflow_runner_for_flavor("single", FAKE_PROVIDER, self.workdir)


# --------------------------------------------------------------------------- #
# run_pdr worker-sequence form (the seam the arms depend on)
# --------------------------------------------------------------------------- #
def _fake_agent(name: str, reply: str) -> Agent:
    def generate(visible: list[Message]) -> Message:
        del visible
        return assistant_message(reply, sender=name, target="user", kind="final")

    return Agent(name, generate, role=f"{name} role")


class PdrSequenceTest(unittest.TestCase):
    def test_sequence_of_workers_is_the_attempt_pool(self) -> None:
        attempts = [_fake_agent("a0", "A0"), _fake_agent("a1", "A1")]
        distiller = _fake_agent("distiller", "BRIEF")
        finalizer = _fake_agent("finalizer", "FINAL")

        result = run_pdr(attempts, distiller, "q", rounds=1, finalizer=finalizer)

        roles = [step.role for step in result.steps]
        self.assertEqual(roles.count("worker"), 2)  # len(attempts), not width
        self.assertEqual(roles.count("distiller"), 1)
        self.assertEqual(roles.count("finalizer"), 1)
        self.assertEqual(result.output, "FINAL")

    def test_finalizer_defaults_to_first_attempt(self) -> None:
        attempts = [_fake_agent("a0", "A0"), _fake_agent("a1", "A1")]
        distiller = _fake_agent("distiller", "BRIEF")

        result = run_pdr(attempts, distiller, "q", rounds=1)

        # No explicit finalizer -> first attempt agent writes the final answer.
        self.assertEqual(result.output, "A0")

    def test_empty_worker_sequence_raises(self) -> None:
        distiller = _fake_agent("distiller", "BRIEF")
        with self.assertRaises(ValueError):
            run_pdr([], distiller, "q", rounds=1)


class SubAgentTraceTest(unittest.TestCase):
    """The facade must persist each sub-agent's full trace (inputs + outputs)."""

    def setUp(self) -> None:
        self.workdir = _init_repo()
        self.store = Path(tempfile.mkdtemp(prefix="sal-store-"))

    def tearDown(self) -> None:
        for d in (self.workdir, self.store):
            subprocess.run(["rm", "-rf", str(d)], check=False)

    def _sub_traces(self) -> list[dict]:
        out = []
        for p in sorted((self.store / "out" / "sub").glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out

    def test_loop_arm_writes_one_sub_trace_with_io(self) -> None:
        with _envs(
            {
                AGENT_FLAVOR_ENV: "loop",
                "SAL_STORE_ROOT": str(self.store),
                config.LOOP_MAX_TURNS.name: "1",
                config.WORKER_MAX_TURNS.name: "2",
            }
        ):
            agent = wc.build_agent(
                provider=FAKE_PROVIDER,
                cwd=self.workdir,
                trace_put=container_store_from_env().put,
            )
            _state, events = agent.run(EDIT_TASK, max_turns=1)
            for _ in events:  # drain so the facade's generate actually runs
                pass
        traces = self._sub_traces()
        # The loop arm resumes ONE State, so it's written once as the full
        # continuation conversation.
        self.assertEqual(len(traces), 1)
        # The trace carries the worker's real messages: the bash tool call and
        # its tool result (the inputs/outputs that were invisible before).
        kinds = {m.get("kind") for t in traces for m in t.get("messages", [])}
        self.assertIn("tool_result", kinds)
        blob = json.dumps(traces[0])
        self.assertIn("bash", blob)

    def test_pdr_writes_a_trace_per_attempt(self) -> None:
        with _envs(
            {
                AGENT_FLAVOR_ENV: "pdr",
                "SAL_STORE_ROOT": str(self.store),
                config.PDR_WIDTH.name: "2",
                config.PDR_ROUNDS.name: "1",
                config.WORKER_MAX_TURNS.name: "3",
            }
        ):
            agent = wc.build_agent(
                provider=FAKE_PROVIDER,
                cwd=self.workdir,
                trace_put=container_store_from_env().put,
            )
            _state, events = agent.run(EDIT_TASK, max_turns=1)
            for _ in events:  # drain so the facade's generate actually runs
                pass
        traces = self._sub_traces()
        # 2 distinct attempt States (each its own worktree) + distiller +
        # finalizer → several distinct sub-traces.
        self.assertGreaterEqual(len(traces), 2)


class ComposeTraceStateTest(unittest.TestCase):
    """The workflow facade's default final trace is a lightweight tree."""

    def setUp(self) -> None:
        self.workdir = _init_repo()
        self.store = Path(tempfile.mkdtemp(prefix="sal-store-"))

    def tearDown(self) -> None:
        for d in (self.workdir, self.store):
            subprocess.run(["rm", "-rf", str(d)], check=False)

    def test_plain_agent_traces_original_state(self) -> None:
        agent = _fake_agent("plain", "ok")
        state = State(task="t")
        self.assertIs(agent.trace_state(state), state)

    def test_builds_one_tool_call_node_per_subagent(self) -> None:
        from simple_agent_lab.trace import run_trace_from_state, trace_record

        overview = [
            {
                "index": 0,
                "label": "attempt",
                "role": "attempt",
                "name": "a0",
                "model": "deepseek/x",
                "tokens": 1234,
                "output": "attempt 0",
                "subpath": "sub/00_attempt.jsonl",
            },
            {
                "index": 1,
                "label": "finalizer",
                "role": "finalizer",
                "name": "fin",
                "model": "deepseek/x",
                "tokens": 99,
                "output": "final patch",
                "subpath": "sub/01_finalizer.jsonl",
            },
        ]

        from simple_agent_lab.workflow import compose_workflow_trace_state

        composed = compose_workflow_trace_state(
            State(task="Solve it."),
            overview=overview,
            final_output="Final answer.",
            agent_name=wc.AGENT_NAME,
        )
        self.assertIsNotNone(composed)
        rec = trace_record(
            run_trace_from_state(
                state=composed, trace_id="pdr.overview", producer="t", meta={}
            )
        )
        tool_spans = [s for s in rec["spans"] if s["kind"] == "tool_call"]
        self.assertEqual(len(tool_spans), 2)  # one per sub-agent, not one big span
        self.assertEqual(
            [s["attributes"]["tool_name"] for s in tool_spans],
            ["attempt", "finalizer"],
        )
        # The node summary points at the drill-down file (no embedded sub_events).
        blob = json.dumps(rec)
        self.assertIn("out/sub/00_attempt.jsonl", blob)
        self.assertNotIn("sub_events", blob)
        # Tiny: a structural tree, not the re-embedded sub-agent logs.
        self.assertLess(len(blob), 50_000)

    def test_workflow_agent_trace_state_uses_subagent_overview(self) -> None:
        from simple_agent_lab.trace import run_trace_from_state, trace_record

        with _envs(
            {
                AGENT_FLAVOR_ENV: "loop",
                "SAL_STORE_ROOT": str(self.store),
                config.LOOP_MAX_TURNS.name: "1",
                config.WORKER_MAX_TURNS.name: "2",
            }
        ):
            agent = wc.build_agent(
                provider=FAKE_PROVIDER,
                cwd=self.workdir,
                trace_put=container_store_from_env().put,
            )
            state, events = agent.run(EDIT_TASK, max_turns=1)
            for _ in events:
                pass

        composed = agent.trace_state(state)
        rec = trace_record(
            run_trace_from_state(
                state=composed, trace_id="loop.overview", producer="t", meta={}
            )
        )
        tool_spans = [s for s in rec["spans"] if s["kind"] == "tool_call"]
        self.assertTrue(tool_spans)


class _envs:
    """Context manager that sets env vars and restores them on exit."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self._prev: dict[str, str | None] = {}

    def __enter__(self) -> "_envs":
        import os

        for key, value in self.values.items():
            self._prev[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc: object) -> None:
        import os

        for key, prev in self._prev.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


if __name__ == "__main__":
    unittest.main()
