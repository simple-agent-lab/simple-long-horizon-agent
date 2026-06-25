# Source-Tree Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace user-facing wrapper-package evolution with DGM-style source-tree evolution over `src/simple_agent_lab/**`.

**Architecture:** Add a framework-level source-tree surface and candidate-source staging contract, then migrate simple, DGM, and AHE-facing recipe behavior onto that surface. Candidate generation uses isolated git/source trees and an agentic meta-strategy; candidate rollout must import the staged candidate `src/simple_agent_lab` before the in-container runner imports the installed baseline package.

**Tech Stack:** Python 3.10+, `unittest`, `uv`, Docker SWE-bench backend, git worktrees or copied source trees, existing `simple_agent_lab.evolution` substrate.

---

## File Structure

- `src/simple_agent_lab/evolution/source_tree.py`
  - New source-tree surface, candidate source artifact helpers, diff validation, cheap validation helpers.
- `src/simple_agent_lab/evolution/surface.py`
  - Keep generic `AgentSurface`; remove user-facing dependence on `python_agent_surface` from real recipes after migrations.
- `src/simple_agent_lab/evals/protocols.py`
  - Add source-tree artifact constants and `RunSpec.pythonpath`.
- `src/simple_agent_lab/evals/bootstrap.py`
  - Add `extra_pythonpath` support before the runner process starts.
- `src/simple_agent_lab/evals/runner.py`
  - Thread `pythonpath` from rollout/run kwargs into `RunSpec` and bootstrap command.
- `src/simple_agent_lab/evolution/components/rollout.py`
  - Support surfaces that stage many candidate source files and request candidate `PYTHONPATH`.
- `src/simple_agent_lab/evolution/components/repo_strategy.py`
  - New agentic source-tree meta-strategy. It creates a candidate tree, runs a coding agent there, validates, and returns a `Proposal`.
- `recipes/simple/evolve.py`
  - Register source-tree surface and agentic repo strategy; stop registering wrapper package as the real simple surface.
- `recipes/dgm/evolve.py`, `recipes/dgm/swebench.py`
  - Use source-tree surface and source diagnostics instead of wrapper package diagnostics for DGM user-facing runs.
- `recipes/ahe/*`
  - Replace wrapper-specific surface factories with `source_tree_surface` for user-facing AHE flows.
- `configs/simple_swebench.yaml`, `configs/dgm_swebench.yaml`, AHE configs
  - Migrate `surface.name` to the source-tree surface.
- `docs/agent-native/self-evolving.md`, `src/simple_agent_lab/evolution/README.md`, recipe READMEs
  - Replace wrapper wording with source-tree wording.
- Tests:
  - `tests/unit/test_evolution_source_tree.py`
  - `tests/unit/test_evals_bootstrap.py`
  - `tests/unit/test_evals_framework.py`
  - `tests/unit/test_swebench_self_evolving_registry.py`
  - existing recipe tests that mention `python_agent_package`.

## Task 1: Add Candidate Pythonpath Support Before Runner Import

**Files:**
- Modify: `src/simple_agent_lab/evals/protocols.py`
- Modify: `src/simple_agent_lab/evals/bootstrap.py`
- Modify: `src/simple_agent_lab/evals/runner.py`
- Test: `tests/unit/test_evals_bootstrap.py`
- Test: `tests/unit/test_evals_framework.py`

- [ ] **Step 1: Write failing bootstrap test**

Add this test to `tests/unit/test_evals_bootstrap.py`:

```python
def test_bootstrap_prepends_candidate_pythonpath_before_runner() -> None:
    script = bootstrap_script(
        runner_argv=("-m", "simple_agent_lab.evals.in_container"),
        install=False,
        extra_pythonpath=("/agent/run/input/source_tree/src",),
    )

    assert 'export PYTHONPATH="/agent/run/input/source_tree/src${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert script.index("export PYTHONPATH=") < script.index('"$AGENT_PYTHON" -m')
```

- [ ] **Step 2: Run failing bootstrap test**

Run:

```bash
uv run python -m unittest tests.unit.test_evals_bootstrap.EvalBootstrapTest.test_bootstrap_prepends_candidate_pythonpath_before_runner
```

Expected: fail because `bootstrap_script` does not accept `extra_pythonpath`.

- [ ] **Step 3: Implement bootstrap pythonpath support**

Modify `src/simple_agent_lab/evals/bootstrap.py`:

```python
def _pythonpath_line(extra_pythonpath: tuple[str, ...]) -> str:
    if not extra_pythonpath:
        return ""
    joined = ":".join(shlex.quote(part) for part in extra_pythonpath)
    return f'export PYTHONPATH="{joined}${{PYTHONPATH:+:$PYTHONPATH}}"'


def bootstrap_script(
    *,
    runner_argv: tuple[str, ...],
    install: bool = True,
    wheelhouse_mount: str | None = None,
    package_extras: tuple[str, ...] = (),
    extra_pythonpath: tuple[str, ...] = (),
) -> str:
    parts = [_PYTHON_SETUP]
    if install:
        parts.append(_install_line(wheelhouse_mount, package_extras))
    if line := _pythonpath_line(extra_pythonpath):
        parts.append(line)
    quoted = " ".join(shlex.quote(part) for part in runner_argv)
    parts.append(f'"$AGENT_PYTHON" {quoted}')
    return "\n".join(parts)
```

- [ ] **Step 4: Write failing runner test**

Add this test near existing `build_command` tests in `tests/unit/test_evals_framework.py`:

```python
def test_build_command_threads_candidate_pythonpath(self) -> None:
    spec = RunSpec(
        suite_name="demo",
        container_module="demo.container",
        instance_id="i1",
        launch_spec=LaunchSpec(image="demo:latest", workdir="/work"),
        max_turns=3,
        provider="fake",
        api_kind="fake",
        pythonpath=("/agent/run/input/source_tree/src",),
    )

    command = build_command(spec)
    script = command[-1]

    self.assertIn(
        'export PYTHONPATH="/agent/run/input/source_tree/src${PYTHONPATH:+:$PYTHONPATH}"',
        script,
    )
```

- [ ] **Step 5: Run failing runner test**

Run:

```bash
uv run python -m unittest tests.unit.test_evals_framework.OrchestrationTest.test_build_command_threads_candidate_pythonpath
```

Expected: fail because `RunSpec` has no `pythonpath` field.

- [ ] **Step 6: Add `RunSpec.pythonpath` and thread it into bootstrap**

Modify `src/simple_agent_lab/evals/protocols.py`:

```python
@dataclass(frozen=True)
class RunSpec:
    ...
    run_name: str = ""
    pythonpath: tuple[str, ...] = ()
```

Modify `src/simple_agent_lab/evals/runner.py`:

```python
def build_command(spec: RunSpec) -> tuple[str, ...]:
    ...
    script = bootstrap_script(
        runner_argv=runner_argv,
        install=spec.install,
        wheelhouse_mount=spec.wheelhouse_mount,
        package_extras=spec.package_extras,
        extra_pythonpath=spec.pythonpath,
    )
    return tuple(spec.launch_spec.shell) + (script,)
```

- [ ] **Step 7: Add `run_suite_instance(..., pythonpath=())`**

Modify the signature in `src/simple_agent_lab/evals/runner.py`:

```python
def run_suite_instance(
    *,
    ...
    extra_artifacts: Mapping[str, bytes] | None = None,
    pythonpath: tuple[str, ...] = (),
) -> RunArtifacts:
```

Thread it into `RunSpec`:

```python
spec = RunSpec(
    ...
    run_name=name or container_name(...),
    pythonpath=tuple(pythonpath),
)
```

- [ ] **Step 8: Run Task 1 tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evals_bootstrap tests.unit.test_evals_framework
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/simple_agent_lab/evals/protocols.py src/simple_agent_lab/evals/bootstrap.py src/simple_agent_lab/evals/runner.py tests/unit/test_evals_bootstrap.py tests/unit/test_evals_framework.py
git commit -m "feat: support candidate source pythonpath"
```

## Task 2: Add Source-Tree Surface and Artifact Staging

**Files:**
- Create: `src/simple_agent_lab/evolution/source_tree.py`
- Modify: `src/simple_agent_lab/evolution/__init__.py`
- Test: `tests/unit/test_evolution_source_tree.py`

- [ ] **Step 1: Write failing source-tree surface tests**

Create `tests/unit/test_evolution_source_tree.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    candidate_source_artifacts,
    source_tree_surface,
    validate_source_tree_edits,
)


class SourceTreeSurfaceTest(unittest.TestCase):
    def test_surface_exposes_only_simple_agent_lab_source(self) -> None:
        surface = source_tree_surface()

        self.assertEqual(surface.id, "source_tree")
        self.assertEqual(surface.entrypoint, "src/simple_agent_lab:package")
        self.assertIn("src/simple_agent_lab/**", surface.component("everything").paths)

    def test_validation_accepts_only_source_paths(self) -> None:
        edits = {
            "src/simple_agent_lab/agents/example.py": "VALUE = 1\n",
            "recipes/simple/evolve.py": "bad\n",
            "src/simple_agent_lab/bad.py": "def nope(:\n",
        }

        valid = validate_source_tree_edits(edits)

        self.assertEqual(
            valid.edits,
            {"src/simple_agent_lab/agents/example.py": "VALUE = 1\n"},
        )
        self.assertEqual(
            valid.rejected,
            ("recipes/simple/evolve.py", "src/simple_agent_lab/bad.py"),
        )

    def test_candidate_source_artifacts_stage_under_input_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            version = store.stage(
                ws,
                base=None,
                edits={
                    "src/simple_agent_lab/__init__.py": "MARKER = 'candidate'\n",
                    "README.md": "ignored\n",
                },
            )

            artifacts = candidate_source_artifacts(version)

        self.assertEqual(
            artifacts[
                "input/source_tree/src/simple_agent_lab/__init__.py"
            ],
            b"MARKER = 'candidate'\n",
        )
        self.assertNotIn("input/source_tree/README.md", artifacts)
        self.assertEqual(
            CANDIDATE_SOURCE_CONTAINER_SRC,
            "/agent/run/input/source_tree/src",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing source-tree tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_source_tree
```

Expected: fail because `simple_agent_lab.evolution.source_tree` does not exist.

- [ ] **Step 3: Implement `source_tree.py`**

Create `src/simple_agent_lab/evolution/source_tree.py`:

```python
"""Source-tree evolution surface for DGM-style repo self-improvement."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from simple_agent_lab.evolution.surface import AgentSurface, SurfaceComponent, ValidatedEdits
from simple_agent_lab.evolution.types import Version

SOURCE_ROOT = "src/simple_agent_lab"
SOURCE_GLOB = f"{SOURCE_ROOT}/**"
SOURCE_TREE_INPUT_ROOT = "input/source_tree"
CANDIDATE_SOURCE_CONTAINER_SRC = "/agent/run/input/source_tree/src"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    output: str = ""


def source_tree_surface() -> AgentSurface:
    return AgentSurface(
        id="source_tree",
        name="Simple Agent Lab source tree",
        description="The real Simple Agent Lab package source used by task agents.",
        entrypoint=f"{SOURCE_ROOT}:package",
        default_files={},
        artifact_key=SOURCE_TREE_INPUT_ROOT,
        components=(
            SurfaceComponent(
                id="everything",
                name="Simple Agent Lab source",
                description="All package code under src/simple_agent_lab.",
                paths=(SOURCE_GLOB,),
                validators=("path_allowed", "python_syntax"),
            ),
        ),
    )


def validate_source_tree_edits(
    edits: Mapping[str, str | bytes | None],
) -> ValidatedEdits:
    accepted: dict[str, str | None] = {}
    rejected: list[str] = []
    for raw_path, raw_content in edits.items():
        path = str(raw_path)
        if not _allowed_source_path(path):
            rejected.append(path)
            continue
        if raw_content is None:
            accepted[path] = None
            continue
        if not isinstance(raw_content, str):
            rejected.append(path)
            continue
        if path.endswith(".py") and not _python_ok(raw_content):
            rejected.append(path)
            continue
        accepted[path] = raw_content
    return ValidatedEdits(accepted, tuple(rejected))


def candidate_source_artifacts(version: Version) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for rel in version.files():
        if not _allowed_source_path(rel):
            continue
        data = (version.dir / rel).read_bytes()
        artifacts[f"{SOURCE_TREE_INPUT_ROOT}/{rel}"] = data
    return artifacts


def cheap_validate_source_tree(root: str | Path) -> ValidationResult:
    root_path = Path(root)
    commands = (
        [sys.executable, "-m", "compileall", "-q", "src/simple_agent_lab"],
        [
            sys.executable,
            "-c",
            "import simple_agent_lab; from simple_agent_lab.agents.starter import make_bash_agent",
        ],
    )
    output: list[str] = []
    env = {"PYTHONPATH": str((root_path / "src").resolve())}
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=root_path,
            env={**dict(__import__("os").environ), **env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        output.append("$ " + " ".join(command))
        output.append(proc.stdout)
        if proc.returncode != 0:
            return ValidationResult(False, "\n".join(output).strip())
    return ValidationResult(True, "\n".join(output).strip())


def _allowed_source_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        not pure.is_absolute()
        and ".." not in pure.parts
        and (pure == PurePosixPath(SOURCE_ROOT) or SOURCE_ROOT in path)
        and path.startswith(SOURCE_ROOT + "/")
    )


def _python_ok(content: str) -> bool:
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True
```

- [ ] **Step 4: Export source-tree helpers**

Modify `src/simple_agent_lab/evolution/__init__.py` to export:

```python
from .source_tree import source_tree_surface

__all__ = [
    ...
    "source_tree_surface",
]
```

- [ ] **Step 5: Run source-tree tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_source_tree
```

Expected: pass.

- [ ] **Step 6: Run format check**

```bash
uv run ruff format --check src/simple_agent_lab/evolution/source_tree.py tests/unit/test_evolution_source_tree.py
```

Expected: files already formatted or formatted cleanly after `uv run ruff format ...`.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/simple_agent_lab/evolution/source_tree.py src/simple_agent_lab/evolution/__init__.py tests/unit/test_evolution_source_tree.py
git commit -m "feat: add source-tree evolution surface"
```

## Task 3: Thread Source-Tree Artifacts Through Rollout

**Files:**
- Modify: `src/simple_agent_lab/evolution/components/rollout.py`
- Test: `tests/unit/test_evolution_rollout.py`

- [ ] **Step 1: Write failing rollout test for candidate source pythonpath**

Add this test to `tests/unit/test_evolution_rollout.py`:

```python
def test_source_tree_artifacts_request_candidate_pythonpath(self) -> None:
    seen: dict[str, object] = {}

    def on_run(spec: RunSpec, bound) -> None:
        seen["pythonpath"] = spec.pythonpath
        seen["candidate_file"] = bound.get(
            "input/source_tree/src/simple_agent_lab/__init__.py"
        )
        bound.put(TRACE_KEY, b'{"events": []}\n')
        bound.put(RESULT_KEY, b'{"reward": 1.0}\n')

    version = store.stage(
        self.ws,
        base=None,
        edits={"src/simple_agent_lab/__init__.py": "MARKER = 1\n"},
    )
    rollout = dataset_rollout(
        suite=_DemoSuite(),
        backend=FakeBackend(on_run=on_run),
        store=LocalDirStore(self.ws / "runs"),
        runs_root=self.ws / "runs",
        version_artifacts=candidate_source_artifacts,
        candidate_pythonpath=("/agent/run/input/source_tree/src",),
    )

    rollout(version, Slice("demo", ({"instance_id": "i1"},)))

    self.assertEqual(seen["pythonpath"], ("/agent/run/input/source_tree/src",))
    self.assertEqual(seen["candidate_file"], b"MARKER = 1\n")
```

Also import:

```python
from simple_agent_lab.evolution.source_tree import candidate_source_artifacts
```

- [ ] **Step 2: Run failing rollout test**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_rollout.RolloutTest.test_source_tree_artifacts_request_candidate_pythonpath
```

Expected: fail because `dataset_rollout` does not accept `candidate_pythonpath`.

- [ ] **Step 3: Add `candidate_pythonpath` to rollout**

Modify `src/simple_agent_lab/evolution/components/rollout.py`:

```python
def dataset_rollout(
    *,
    suite: Suite,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
    version_artifacts: Callable[[Version], Mapping[str, bytes]] | None = None,
    candidate_pythonpath: Sequence[str] = (),
) -> Rollout:
```

Inside the `run_dataset(...)` call, add:

```python
pythonpath=tuple(candidate_pythonpath),
```

Then update `rollout_from_suite(...)` to accept and forward:

```python
version_artifacts: Callable[[Version], Mapping[str, bytes]] | None = None,
candidate_pythonpath: Sequence[str] = (),
```

Inside `rollout_from_suite(...)`, pass the explicit artifact hook when present:

```python
return dataset_rollout(
    suite=suite,
    backend=backend,
    store=store,
    runs_root=runs_root,
    concurrency=concurrency,
    run_kwargs=run_kwargs,
    version_artifacts=version_artifacts or surface.artifacts_from_version,
    candidate_pythonpath=candidate_pythonpath,
)
```

- [ ] **Step 4: Add `pythonpath` to dataset runner**

Modify `src/simple_agent_lab/evals/dataset.py` so `run_dataset(...)` accepts `pythonpath: tuple[str, ...] = ()` and passes it through to `run_suite_instance(...)`.

Use this exact signature addition:

```python
pythonpath: tuple[str, ...] = (),
```

And pass:

```python
pythonpath=pythonpath,
```

- [ ] **Step 5: Run rollout tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_rollout tests.unit.test_evals_framework
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/simple_agent_lab/evolution/components/rollout.py src/simple_agent_lab/evals/dataset.py tests/unit/test_evolution_rollout.py
git commit -m "feat: stage candidate source for rollout"
```

## Task 4: Add Agentic Source-Tree Meta-Strategy

**Files:**
- Create: `src/simple_agent_lab/evolution/components/repo_strategy.py`
- Modify: `src/simple_agent_lab/evolution/registry.py`
- Test: `tests/unit/test_evolution_repo_strategy.py`

- [ ] **Step 1: Write failing tests for diff-to-proposal and validation**

Create `tests/unit/test_evolution_repo_strategy.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.components.repo_strategy import (
    proposal_from_candidate_tree,
)


class RepoStrategyTest(unittest.TestCase):
    def test_proposal_contains_only_source_tree_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            changed = root / "changed"
            (base / "src/simple_agent_lab").mkdir(parents=True)
            (changed / "src/simple_agent_lab").mkdir(parents=True)
            (base / "src/simple_agent_lab/__init__.py").write_text("A = 1\n")
            (changed / "src/simple_agent_lab/__init__.py").write_text("A = 2\n")
            (changed / "recipes").mkdir()
            (changed / "recipes/ignored.py").write_text("bad\n")

            proposal = proposal_from_candidate_tree(
                base,
                changed,
                base_hash="abc123abc123",
                note="change marker",
                evidence=("local validation passed",),
            )

        self.assertEqual(
            proposal.edits,
            {"src/simple_agent_lab/__init__.py": "A = 2\n"},
        )
        self.assertEqual(proposal.base, "abc123abc123")
        self.assertEqual(proposal.kind, "source")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing repo strategy tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_repo_strategy
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement minimal repo strategy helpers**

Create `src/simple_agent_lab/evolution/components/repo_strategy.py`:

```python
"""Agentic source-tree meta-strategy helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.core import Agent
from simple_agent_lab.evolution.components.strategy import content_changing_edits
from simple_agent_lab.evolution.source_tree import (
    cheap_validate_source_tree,
    validate_source_tree_edits,
)
from simple_agent_lab.evolution.types import Context, Proposal, Version
from simple_agent_lab.llm import Provider


def proposal_from_candidate_tree(
    base_tree: Path,
    changed_tree: Path,
    *,
    base_hash: str,
    note: str,
    evidence: Sequence[str] = (),
) -> Proposal:
    edits: dict[str, str | bytes | None] = {}
    base_files = _source_files(base_tree)
    changed_files = _source_files(changed_tree)
    for rel, changed_path in changed_files.items():
        data = changed_path.read_bytes()
        try:
            value: str | bytes = data.decode("utf-8")
        except UnicodeDecodeError:
            value = data
        base_path = base_files.get(rel)
        if base_path is None or base_path.read_bytes() != data:
            edits[rel] = value
    for rel in base_files:
        if rel not in changed_files:
            edits[rel] = None
    valid = validate_source_tree_edits(edits)
    return Proposal(
        base=base_hash,
        edits=valid.edits,
        note=note,
        evidence=tuple(evidence)
        + tuple(f"discarded-disallowed-path:{path}" for path in valid.rejected),
        kind="source",
    )


def source_tree_agent_strategy(
    *,
    provider: Provider,
    repo_root: str | Path,
    max_turns: int = 20,
    validation: Callable[[Path], Any] = cheap_validate_source_tree,
    parent_selection: str = "current",
    parent_selector: Callable[[Context, str], str] | None = None,
) -> Callable[[Context], Proposal | None]:
    repo_root = Path(repo_root).resolve()

    def strategy(ctx: Context) -> Proposal | None:
        base = _select_parent(ctx, parent_selection, parent_selector)
        with tempfile.TemporaryDirectory(prefix="sal-source-candidate-") as tmp:
            candidate = Path(tmp) / "repo"
            _copy_repo_source(repo_root, candidate)
            agent = _build_meta_agent(provider=provider, cwd=candidate)
            task = _meta_task(base, ctx.failures)
            agent.run(task, max_turns=max_turns)
            result = validation(candidate)
            if not result.ok:
                return Proposal(
                    base=base.hash,
                    edits={},
                    note="candidate failed cheap source validation",
                    evidence=(result.output,),
                    kind="source-invalid",
                )
            proposal = proposal_from_candidate_tree(
                repo_root,
                candidate,
                base_hash=base.hash,
                note="agentic source-tree edit",
                evidence=("cheap validation passed",),
            )
            edits, unchanged = content_changing_edits(base, proposal.edits)
            if not edits:
                return None
            return Proposal(
                base=proposal.base,
                edits=edits,
                note=proposal.note,
                evidence=proposal.evidence
                + tuple(f"discarded-unchanged-path:{path}" for path in unchanged),
                kind=proposal.kind,
            )

    return strategy


def _select_parent(
    ctx: Context,
    parent_selection: str,
    parent_selector: Callable[[Context, str], str] | None,
) -> Version:
    if parent_selection == "current":
        return ctx.current
    if parent_selector is None:
        raise ValueError(
            "non-current parent selection requires a recipe-provided parent_selector"
        )
    selected = parent_selector(ctx, parent_selection) or ctx.current.hash
    return ctx.version(selected)


def _source_files(root: Path) -> dict[str, Path]:
    src_root = root / "src/simple_agent_lab"
    if not src_root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(src_root.rglob("*"))
        if path.is_file()
    }


def _copy_repo_source(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        "evals/out",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    )
    shutil.copytree(src, dst, ignore=ignore)


def _build_meta_agent(*, provider: Provider, cwd: Path) -> Agent:
    return make_bash_agent(
        provider=provider,
        cwd=cwd,
        name="source_tree_meta_agent",
        system_prompt=(
            "You improve Simple Agent Lab by editing only files under "
            "src/simple_agent_lab/. Inspect first, make one focused change, "
            "run python -m compileall -q src/simple_agent_lab, and stop."
        ),
    )


def _meta_task(base: Version, failures: Sequence[Any]) -> str:
    failure_lines = "\n".join(
        f"- {run.instance_id}: reward={run.reward}" for run in failures
    ) or "- none"
    return (
        f"Current version: {base.hash}\n"
        f"Recent failing runs:\n{failure_lines}\n\n"
        "Make one focused source change under src/simple_agent_lab/ likely to "
        "improve benchmark reward. Do not edit recipes, configs, tests, docs, "
        "outputs, or secrets."
    )
```

- [ ] **Step 4: Register strategy**

Modify `src/simple_agent_lab/evolution/registry.py`:

```python
from simple_agent_lab.evolution.components.repo_strategy import source_tree_agent_strategy

STRATEGIES = {
    ...
    "source_tree_agent": source_tree_agent_strategy,
}
```

- [ ] **Step 5: Run repo strategy tests**

Run:

```bash
uv run python -m unittest tests.unit.test_evolution_repo_strategy
```

Expected: pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/simple_agent_lab/evolution/components/repo_strategy.py src/simple_agent_lab/evolution/registry.py tests/unit/test_evolution_repo_strategy.py
git commit -m "feat: add agentic source-tree strategy"
```

## Task 5: Migrate Simple Recipe to Source-Tree Evolution

**Files:**
- Modify: `recipes/simple/evolve.py`
- Modify: `configs/simple_swebench.yaml`
- Test: `tests/unit/test_swebench_self_evolving_registry.py`
- Test: `tests/unit/test_recipes_smoke.py`

- [ ] **Step 1: Update registry test expectation**

Modify `tests/unit/test_swebench_self_evolving_registry.py` so the recipe test asserts:

```python
self.assertIn("source_tree", registry.SURFACES)
self.assertNotIn("python_agent_package", registry.SURFACES)
self.assertIn("source_tree_agent", registry.STRATEGIES)
```

Run:

```bash
uv run python -m unittest tests.unit.test_swebench_self_evolving_registry
```

Expected: fail because simple still registers `python_agent_package`.

- [ ] **Step 2: Register source-tree surface in simple recipe**

Modify `recipes/simple/evolve.py` imports:

```python
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    candidate_source_artifacts,
    source_tree_surface,
)
from simple_agent_lab.evolution.components.repo_strategy import source_tree_agent_strategy
```

Replace the `python_agent_package` registration with:

```python
registry.SURFACES.setdefault("source_tree", lambda **_args: source_tree_surface())
registry.STRATEGIES.setdefault("source_tree_agent", source_tree_agent_strategy)
```

Keep `model_program` registered only if tests or non-user fixtures still need it. Do not use it in `configs/simple_swebench.yaml`.

- [ ] **Step 3: Wire rollout source artifacts in configured builder**

Modify `src/simple_agent_lab/evolution/config.py` so `build_self_evolving_run(...)` detects the source-tree surface and passes source artifacts plus candidate `PYTHONPATH` into `rollout_from_suite(...)`.

Add imports:

```python
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    candidate_source_artifacts,
)
```

Before the `rollout_from_suite(...)` call, add:

```python
version_artifacts = None
candidate_pythonpath: tuple[str, ...] = ()
if surface.id == "source_tree":
    version_artifacts = candidate_source_artifacts
    candidate_pythonpath = (CANDIDATE_SOURCE_CONTAINER_SRC,)
```

Pass both into `rollout_from_suite(...)`:

```python
rollout = rollout_from_suite(
    suite=suite,
    surface=surface,
    backend=backend,
    store=store,
    runs_root=run_root / "runs",
    concurrency=config.execution.parallel,
    run_kwargs={
        "max_turns": config.execution.max_turns,
        **dict(config.execution.run_kwargs),
    },
    version_artifacts=version_artifacts,
    candidate_pythonpath=candidate_pythonpath,
)
```

- [ ] **Step 4: Update `configs/simple_swebench.yaml`**

Change:

```yaml
surface:
  name: source_tree
  editable_components: [everything]
  artifact_key: input/source_tree
  default: simple_agent_lab_source
strategy:
  name: source_tree_agent
```

Remove wrapper-specific `agent_package` wording from the strategy prompt.

- [ ] **Step 5: Run simple recipe tests**

Run:

```bash
uv run python -m unittest tests.unit.test_swebench_self_evolving_registry tests.unit.test_recipes_smoke
```

Expected: pass after updating assertions that printed plans show `surface: source_tree`.

- [ ] **Step 6: Commit Task 5**

```bash
git add recipes/simple/evolve.py configs/simple_swebench.yaml tests/unit/test_swebench_self_evolving_registry.py tests/unit/test_recipes_smoke.py
git commit -m "feat: migrate simple recipe to source evolution"
```

## Task 6: Migrate DGM Recipe Diagnostics and Surface

**Files:**
- Modify: `recipes/dgm/evolve.py`
- Modify: `recipes/dgm/swebench.py`
- Modify: `configs/dgm_swebench.yaml`
- Test: `tests/unit/test_swebench_evolving_rollout.py`
- Test: `tests/unit/test_dgm_swebench_diagnostics.py`
- Test: `tests/unit/test_recipes_smoke.py`

- [ ] **Step 1: Replace wrapper scaffold seed tests**

Update tests that assert `dgm_default_agent_package()` loads. Replace with a source-tree seed/artifact test:

```python
def test_dgm_uses_source_tree_artifacts(self):
    self.assertEqual(er.EVOLVING_SOURCE_ROOT, "src/simple_agent_lab")
    self.assertEqual(er.CANDIDATE_SOURCE_CONTAINER_SRC, "/agent/run/input/source_tree/src")
```

Run:

```bash
uv run python -m unittest tests.unit.test_swebench_evolving_rollout
```

Expected: fail until DGM constants are updated.

- [ ] **Step 2: Update DGM SWE-bench support**

Modify `recipes/dgm/swebench.py`:

```python
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    SOURCE_ROOT as EVOLVING_SOURCE_ROOT,
    candidate_source_artifacts,
)


def version_source_artifacts(version: Version) -> dict[str, bytes]:
    return candidate_source_artifacts(version)
```

Remove or demote `_AGENT_PROGRAM`, `_PROMPTS`, `_REVIEW`, `_TOOLS`, and wrapper-package seed functions from user-facing DGM paths.

- [ ] **Step 3: Update DGM workflow**

Modify `recipes/dgm/evolve.py`:

```python
from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_SOURCE_CONTAINER_SRC,
    source_tree_surface,
)
from simple_agent_lab.evolution.components.repo_strategy import source_tree_agent_strategy
```

Use:

```python
seed=source_tree_surface().seed_files()
strategy = source_tree_agent_strategy(
    provider=provider,
    repo_root=ROOT,
    parent_selection=args.parent_selection,
    parent_selector=select_archive_parent,
)
```

- [ ] **Step 4: Update DGM rollout version artifacts**

Change the DGM rollout builder call from:

```python
version_artifacts=er.version_package_artifacts
```

to:

```python
version_artifacts=er.version_source_artifacts
candidate_pythonpath=(er.CANDIDATE_SOURCE_CONTAINER_SRC,)
```

- [ ] **Step 5: Update diagnostics names**

Replace `agent_package` diagnostics with `source_tree` diagnostics:

```python
"source_tree": {
    "staged": True,
    "cheap_validation": "passed" | "failed",
    "used_candidate_source": True,
}
```

Update reward/diagnostic tests to reject invalid source candidates instead of wrapper load fallback.

- [ ] **Step 6: Run DGM tests**

Run:

```bash
uv run python -m unittest tests.unit.test_swebench_evolving_rollout tests.unit.test_dgm_swebench_diagnostics tests.unit.test_recipes_smoke
```

Expected: pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add recipes/dgm/evolve.py recipes/dgm/swebench.py configs/dgm_swebench.yaml tests/unit/test_swebench_evolving_rollout.py tests/unit/test_dgm_swebench_diagnostics.py tests/unit/test_recipes_smoke.py
git commit -m "feat: migrate dgm recipe to source evolution"
```

## Task 7: Update AHE and Documentation

**Files:**
- Modify: AHE recipe files under `recipes/ahe/`
- Modify: `configs/ahe_swebench.yaml`
- Modify: `docs/agent-native/self-evolving.md`
- Modify: `src/simple_agent_lab/evolution/README.md`
- Modify: recipe READMEs
- Test: `tests/unit/test_recipes_ahe_surface.py`
- Test: `tests/unit/test_recipes_smoke.py`

- [ ] **Step 1: Search wrapper language**

Run:

```bash
rg -n "agent package|agent/|python_agent_package|wrapper|whole agent program|agent_package" recipes docs configs tests/unit
```

Expected: list of references to migrate. References that stay for unit coverage
must be renamed to `toy_agent_package_surface`.

- [ ] **Step 2: Update AHE to observe source-tree runs**

Replace AHE wrapper-surface wording with source-tree wording. Any AHE user-facing surface factory must return `source_tree_surface()`. Rename old wrapper scaffolds kept for unit coverage to `toy_agent_package_surface` and keep them out of configs and recipe defaults.

- [ ] **Step 3: Update docs**

In `docs/agent-native/self-evolving.md`, replace:

```text
Both evolve the whole agent program under `agent/`
```

with:

```text
User-facing SWE-bench recipes evolve the real Simple Agent Lab package source
under `src/simple_agent_lab/**`. The earlier wrapper-package surface is kept
only as a test fixture named `toy_agent_package_surface` and is not a real
self-evolution recipe.
```

In `src/simple_agent_lab/evolution/README.md`, change the configured YAML example to:

```yaml
surface:
  name: source_tree
  editable_components: [everything]
  artifact_key: input/source_tree
  default: simple_agent_lab_source
```

- [ ] **Step 4: Run docs/reference tests**

Run:

```bash
uv run python -m unittest tests.unit.test_recipes_ahe_surface tests.unit.test_recipes_smoke
uv run python scripts/lint_docs.py
```

Expected: pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add recipes/ahe configs/ahe_swebench.yaml docs/agent-native/self-evolving.md src/simple_agent_lab/evolution/README.md recipes tests/unit/test_recipes_ahe_surface.py tests/unit/test_recipes_smoke.py
git commit -m "docs: describe source-tree evolution recipes"
```

## Task 8: Add Synthetic Candidate-Source Smoke

**Files:**
- Create: `tests/unit/test_source_tree_rollout_smoke.py`

- [ ] **Step 1: Write synthetic import-precedence test**

Create `tests/unit/test_source_tree_rollout_smoke.py`:

```python
from __future__ import annotations

import unittest

from simple_agent_lab.evals.protocols import LaunchSpec, RunSpec
from simple_agent_lab.evals.runner import build_command


class SourceTreeRolloutSmokeTest(unittest.TestCase):
    def test_candidate_pythonpath_precedes_in_container_runner(self) -> None:
        spec = RunSpec(
            suite_name="demo",
            container_module="demo.container",
            instance_id="i1",
            launch_spec=LaunchSpec(image="demo:latest", workdir="/work"),
            max_turns=3,
            provider="fake",
            api_kind="fake",
            pythonpath=("/agent/run/input/source_tree/src",),
        )

        command = build_command(spec)
        script = command[-1]

        self.assertIn(
            'export PYTHONPATH="/agent/run/input/source_tree/src${PYTHONPATH:+:$PYTHONPATH}"',
            script,
        )
        self.assertLess(
            script.index("export PYTHONPATH="),
            script.index("simple_agent_lab.evals.in_container"),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run synthetic smoke**

Run:

```bash
uv run python -m unittest tests.unit.test_source_tree_rollout_smoke
```

Expected: pass.

- [ ] **Step 3: Commit Task 8**

```bash
git add tests/unit/test_source_tree_rollout_smoke.py
git commit -m "test: verify candidate source import precedence"
```

## Task 9: Final Verification and Tiny Real Smoke

**Files:**
- No source changes unless verification exposes a bug.

- [ ] **Step 1: Run focused unit suite**

Run:

```bash
uv run python -m unittest \
  tests.unit.test_evolution_source_tree \
  tests.unit.test_evolution_repo_strategy \
  tests.unit.test_evolution_rollout \
  tests.unit.test_evals_bootstrap \
  tests.unit.test_evals_framework \
  tests.unit.test_swebench_self_evolving_registry \
  tests.unit.test_swebench_evolving_rollout \
  tests.unit.test_dgm_swebench_diagnostics \
  tests.unit.test_recipes_smoke
```

Expected: all tests pass.

- [ ] **Step 2: Run format check**

Run:

```bash
uv run ruff format --check \
  src/simple_agent_lab/evolution/source_tree.py \
  src/simple_agent_lab/evolution/components/repo_strategy.py \
  src/simple_agent_lab/evals/protocols.py \
  src/simple_agent_lab/evals/bootstrap.py \
  src/simple_agent_lab/evals/runner.py \
  src/simple_agent_lab/evolution/components/rollout.py \
  recipes/simple/evolve.py \
  recipes/dgm/evolve.py \
  recipes/dgm/swebench.py \
  tests/unit/test_evolution_source_tree.py \
  tests/unit/test_evolution_repo_strategy.py
```

Expected: files already formatted.

- [ ] **Step 3: Run tiny dry-run**

Run:

```bash
bash runs/run_self_evolving_simple.sh --run-id source-tree-dry --config configs/simple_swebench.yaml
```

Expected output includes:

```text
surface: source_tree
editable components: everything
```

- [ ] **Step 4: Run tiny real source-tree smoke**

Use a one-instance, one-round config equivalent to the prior smoke, but with:

```yaml
surface:
  name: source_tree
strategy:
  name: source_tree_agent
```

Run:

```bash
bash runs/run_self_evolving_simple.sh \
  --run-id simple-source-smoke \
  --execute \
  --config configs/my_simple_swebench_smoke.yaml
```

Expected:

- candidate version contains `src/simple_agent_lab/**` edits or proposal failure diagnostics;
- no `agent/agent_program.py` wrapper candidate is produced;
- cheap validation runs before SWE-bench rollout;
- if a candidate reaches rollout, its container result includes evidence that candidate source was staged.

- [ ] **Step 5: Report final state**

Summarize:

- commits made;
- tests run;
- whether real smoke used actual Docker and provider;
- any known remaining limitations, especially meta-agent quality and whether DGM archive invalid candidates enter the archive or only the decision log.
