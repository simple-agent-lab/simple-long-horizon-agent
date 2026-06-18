# Self-Evolving Framework V1 - Design

- Date: 2026-06-18
- Status: Approved direction; pending written spec review
- Scope: PR37 follow-up design for a clearer self-evolving framework, general
  benchmark execution interface, first-class agent modification surface, and a
  general YAML run config.

## Problem

PR37 has a good substrate in `src/simple_agent_lab/evolution/`: immutable
versions, an append-only decision log, a small loop, swappable rollout/reward/
strategy/criterion components, and runnable simple/DGM recipes.

The layer above the substrate is still too tangled. Simple self-evolution still
depends on SWE-bench-specific adapter glue for concepts that are not actually
SWE-bench-specific: what part of the agent can evolve, how a version becomes
runtime artifacts, how an instance subset is named, and how run configuration is
declared. The result is hard to explain: users see many adapters before they see
the simple idea.

The framework should make self-evolving runs feel like composing a few clear
objects, not threading benchmark-specific helper functions through recipes.

## Goals

1. Keep `Suite` as the universal benchmark interface. SWE-bench must be a normal
   suite, not a special evolution target.
2. Introduce `AgentSurface` as the first-class semantic agent modification
   contract: what agent capabilities can change, which components are editable,
   how edits are validated, and how version files become runtime artifacts.
3. Rename or generalize the current benchmark-instance `Slice` concept into
   `InstanceSet`, so it is not confused with an agent slice/surface.
4. Add a generic `rollout_from_suite(...)` bridge from evolution versions to
   suite execution.
5. Add a general YAML self-evolving run config that declaratively composes the
   public objects while keeping the core Python-first.
6. Keep recipes as examples or presets over the general config path, not as
   bespoke benchmark glue.

## Non-Goals

- No new `Benchmark` class. `Suite` is the benchmark abstraction.
- No `EvolutionHarness` class. `Experiment` plus helper functions is enough.
- No `Candidate` or `RunSubject` in v1. `Version` is the evolution subject;
  `RunSpec` is the backend-facing execution request.
- No benchmark-owned agent surface module. Surfaces belong to evolution/agent
  infrastructure (`src/simple_agent_lab/evolution/surface.py`) and are composed
  with suites by config or recipe registration such as
  `evals/swebench/self_evolving.py`.
- No arbitrary Python import paths in v1 YAML. The first config format should
  use small registries with readable names.

## Core Mental Model

```text
Suite + AgentSurface + InstanceSet + EvolutionConfig
    -> rollout
    -> Runs
    -> reward / criterion
    -> Decision
```

The axes are independent:

- `Suite` answers: what benchmark is being run?
- `AgentSurface` answers: what agent interface and components may evolve?
- `InstanceSet` answers: which benchmark cases are used?
- `EvolutionConfig` answers: how long and by which search policy do we evolve?

## Target Module Shape

```text
src/simple_agent_lab/evals/
  protocols.py          # Suite, LaunchSpec, RunSpec, ContainerBackend, ArtifactStore
  runner.py             # run_suite_instance(...)
  dataset.py            # run_dataset(...)
  instances.py          # InstanceSet

src/simple_agent_lab/evolution/
  surface.py            # AgentSurface, SurfaceComponent
  run_config.py         # general YAML config dataclasses + loader
  run.py                # generic self-evolving CLI / entrypoint
  experiment.py         # Experiment
  types.py              # Version, Run, Proposal, Decision, Verdict
  components/
    rollout.py          # rollout_from_suite(...)
    strategy.py         # model_program_strategy(surface=...)
    reward.py
    criterion.py

evals/swebench/
  suite.py              # SWE-bench host Suite
  evaluate_predictions.py
  ...                   # official scoring / reporting helpers
```

`RunSpec` remains in `evals.protocols` and stays backend-facing. Recipes and
YAML users should not construct it directly.

## Public Concepts

### Suite

`Suite` remains the universal benchmark interface:

```python
class Suite(Protocol):
    name: str
    container_module: str

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec: ...
    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]: ...
    def eval_inputs(
        self, instance: Mapping[str, Any]
    ) -> Mapping[str, Any] | None: ...
```

Responsibilities:

- Resolve per-instance launch requirements as `LaunchSpec`.
- Hide gold/private fields from the agent via `task_input`.
- Optionally stage hidden eval inputs via `eval_inputs`.
- Point to the container half through `container_module`.

Non-responsibilities:

- It does not know evolution.
- It does not know `AgentSurface`.
- It does not own a recipe or YAML config.

### AgentSurface

`AgentSurface` is the first-class editable-agent contract. It is not a path
wrapper. It tells humans and meta-agents what they are modifying, which
components are editable, what entrypoint must remain valid, and how the changed
files become runtime artifacts.

```python
@dataclass(frozen=True)
class SurfaceComponent:
    id: str
    name: str
    description: str
    paths: tuple[str, ...]
    validators: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSurface:
    id: str
    name: str
    description: str
    entrypoint: str
    default_files: Mapping[str, str]
    artifact_key: str
    components: tuple[SurfaceComponent, ...]

    def seed_files(self) -> dict[str, str]: ...
    def component(self, id: str) -> SurfaceComponent: ...
    def validate_edits(
        self,
        edits: Mapping[str, str | None],
        *,
        components: Sequence[str],
    ) -> ValidatedEdits: ...
    def prompt_brief(self, *, components: Sequence[str]) -> str: ...
    def files_from_version(self, version: Version) -> dict[str, str]: ...
    def artifacts_from_version(self, version: Version) -> dict[str, bytes]: ...
```

Responsibilities:

- Define the evolvable agent interface in human-readable terms.
- Define named editable components, such as `prompts`, `tool_policy`,
  `memory_policy`, `agent_program`, and `everything`.
- Map those components to path patterns for enforcement.
- Preserve required contracts such as `agent/agent_program.py:build_agent`.
- Provide initial files for the seed `Version`.
- Validate/filter proposed edits for the selected components.
- Generate a prompt brief that tells the meta-agent what the selected components
  mean, not only where their files live.
- Extract surface files from a `Version`.
- Encode surface files as runtime artifacts for suite execution.

The current hidden surface is spread across `prefix="agent/"`,
`safe_prefix_edits`, `seed_files`, `package_files`, and
`version_package_artifacts`. V1 makes that one explicit semantic concept.

The surface does not execute benchmarks and does not interpret benchmark
results. It only produces staged artifacts. The suite/container still decides
how a staged artifact is used at runtime.

V1 can ship one default surface for the current evolvable Python agent package.
It should include several demo components so users can choose the intended edit
scope without thinking in raw paths:

```text
agent_program
  paths: agent/agent_program.py
  meaning: build_agent entrypoint and agent assembly

prompts
  paths: agent/prompts.py, agent/prompts/**
  meaning: system prompts, task framing, response policy

tool_policy
  paths: agent/tools.py, agent/tool_policy.py, agent/tools/**
  meaning: tool choice, shell usage, retry behavior

memory_policy
  paths: agent/memory.py, agent/memory/**
  meaning: how prior evidence or notes are used

everything
  paths: agent/**
  meaning: unrestricted edits to the whole agent package
```

`everything` preserves the current simple/DGM behavior. The narrower components
give demos and users a clearer way to run prompt-only or policy-only evolution
later.

Validators should be named, built-in checks in v1 rather than a plugin system.
Useful initial validators:

```text
path_allowed       edited paths must match selected component paths
python_syntax      Python file contents must parse
entrypoint_exists  the surface entrypoint contract remains present
```

### InstanceSet

`InstanceSet` names and freezes the benchmark cases for a run or fair
comparison:

```python
@dataclass(frozen=True)
class InstanceSet:
    id: str
    instances: tuple[Mapping[str, Any], ...]

    @property
    def sha(self) -> str: ...

    @property
    def n(self) -> int: ...
```

Responsibilities:

- Name train, heldout, test, or custom case collections.
- Provide a stable hash for run IDs and cache reuse.
- Make fair A/B comparisons explicit: baseline and candidate run on the same
  `InstanceSet`.

The current `Slice` type has this meaning in code, but the name is confusing
because "agent slice" naturally means the editable agent surface. V1 should add
`InstanceSet` and keep `Slice` only as a temporary compatibility alias or thin
wrapper.

### Experiment

`Experiment` remains the slim orchestration object:

```text
current version
-> rollout baseline
-> strategy proposes edits
-> stage candidate version
-> rollout candidate
-> reward + criterion
-> append decision
-> promote if accepted
```

`Experiment` should accept an `InstanceSet` conceptually, even if the first
implementation keeps compatibility with `slice_id` and `instances`.

### RunSpec

`RunSpec` is the compiled one-instance execution request created by
`run_suite_instance(...)` and consumed by a `ContainerBackend`.

It is not part of the recipe authoring surface. Users think in `Suite`,
`InstanceSet`, `AgentSurface`, backend/store, and rollout terms.

## Generic Rollout

`rollout_from_suite(...)` replaces benchmark-specific evolution rollout
builders for the common path:

```python
def rollout_from_suite(
    *,
    suite: Suite,
    surface: AgentSurface,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
) -> Rollout:
    ...
```

The returned function has the evolution rollout shape:

```python
Rollout = Callable[[Version, InstanceSet], Sequence[Run]]
```

Its behavior:

1. Build a deterministic run id from `version.hash` and `instance_set.sha`.
2. Reuse prior results when every wanted instance already has `out/result.json`.
3. Read provider settings from the version's `provider.json`.
4. Ask `surface.artifacts_from_version(version)` for staged runtime artifacts.
5. Call `run_dataset(...)` with the suite, instances, backend, store, and run
   kwargs.
6. Return evolution `Run` views over the run directories.

Benchmark-specific postprocessing, such as SWE-bench official parity reporting,
stays outside the generic rollout as small suite/report helpers.

## Strategy Integration

`model_program_strategy(...)` should take `surface`:

```python
strategy = model_program_strategy(
    provider=provider,
    surface=surface,
    system_prompt=SYSTEM_PROMPT,
)
```

The strategy uses:

- `surface.prompt_brief(components=...)` to explain the editable components to
  the meta-agent.
- `surface.validate_edits(..., components=...)` before returning a `Proposal`.

For compatibility, `prefix=` can remain temporarily, but `surface=` should be
the preferred API. If a legacy `prefix=` is supplied, it should create a simple
single-component surface internally or follow the old path until removed.

## General YAML Config

YAML is a general self-evolving run manifest. It is not recipe-local only.

Principle:

```text
YAML names and configures objects.
Python implements behavior.
Framework objects do not read YAML directly.
```

V1 uses registry names, not arbitrary import paths.

Example:

```yaml
run:
  id: pr37-simple
  output_root: evals/out/simple_swebench
  execute: false
  reset: false
  dotenv: .env

suite:
  name: swebench
  args:
    dataset_name: princeton-nlp/SWE-bench_Verified
    in_env_scoring: true

surface:
  name: python_agent_package
  editable_components:
    - everything
  artifact_key: input/agent_package.json
  default: simple_agent_package

instances:
  train:
    id: swebench-train
    path: data/train.jsonl
  heldout:
    id: swebench-heldout
    path: data/test.jsonl

execution:
  backend:
    name: local_docker
    args:
      wheelhouse: ""
      uv_binary: ""
  store:
    name: local_dir
  parallel: auto
  max_turns: 75

model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
  base_url_env: OPENAI_BASE_URL

strategy:
  name: model_program
  args:
    system_prompt: swebench_agent

evolution:
  algorithm: simple
  rounds: 5
  criterion:
    name: promote_not_worse
    args:
      dim: reward
      tol: 0.0

evaluation:
  baseline_heldout: true
  final_heldout: true
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: true
```

DGM changes the evolution block:

```yaml
evolution:
  algorithm: dgm
  rounds: 4
  branches: 3
  meta_concurrency: 3
  parent_selection: score_child_prop
  criterion:
    name: dgm_admission
```

### Typed Config Values

`src/simple_agent_lab/evolution/run_config.py` should define typed dataclasses:

```python
@dataclass(frozen=True)
class SurfaceConfig:
    name: str
    editable_components: tuple[str, ...]
    default: str
    artifact_key: str


@dataclass(frozen=True)
class SelfEvolvingConfig:
    run: RunConfig
    suite: SuiteConfig
    surface: SurfaceConfig
    instances: InstancesConfig
    execution: ExecutionConfig
    model: ModelConfig
    strategy: StrategyConfig
    evolution: EvolutionRunConfig
    evaluation: EvaluationConfig
```

The loader:

```python
def load_self_evolving_config(path: str | Path) -> SelfEvolvingConfig: ...
```

The builder:

```python
def build_self_evolving_run(config: SelfEvolvingConfig) -> SelfEvolvingRun: ...
```

`SelfEvolvingRun` is a prepared bundle of normal runtime objects, not a new
harness class with policy:

```python
@dataclass(frozen=True)
class SelfEvolvingRun:
    config: SelfEvolvingConfig
    suite: Suite
    surface: AgentSurface
    editable_components: tuple[str, ...]
    train: InstanceSet
    heldout: InstanceSet | None
    rollout: Rollout
    strategy: Strategy
    experiment: Experiment
```

The builder resolves registry names and constructs this bundle:

```text
SelfEvolvingConfig
-> Suite
-> AgentSurface
-> InstanceSet(s)
-> Backend + Store
-> rollout_from_suite
-> Provider + Strategy
-> Experiment
```

## Generic Runner

Add a general CLI entrypoint:

```bash
python -m simple_agent_lab.evolution.run \
  --config configs/simple_swebench.yaml \
  --execute
```

Supported CLI overrides in v1:

```text
--config
--run-id
--execute
--reset
--monitor
```

Other knobs live in YAML. This keeps the CLI small and makes the YAML the
reproducible run manifest.

The runner flow:

```text
load YAML
apply CLI overrides
validate typed config
load dotenv
construct Suite
construct AgentSurface
resolve editable surface components
load train and heldout InstanceSets
construct Backend + Store
construct rollout_from_suite
construct Provider
construct strategy
construct Experiment
optionally run baseline heldout
run selected algorithm
optionally run final heldout
write summary
write resolved_config.json or resolved_config.yaml to the run directory
```

## Registries

V1 config should use small registries:

```text
suite.name: swebench
surface.default: simple_agent_package
surface.name: python_agent_package
surface.editable_components: everything | prompts | tool_policy | memory_policy | agent_program
backend.name: local_docker
store.name: local_dir
strategy.name: model_program
criterion.name: promote_not_worse
evolution.algorithm: simple | dgm
```

The registry values should be explicit and grep-able. Arbitrary import strings
are a future extension, not v1.

## Recipe Authoring

A user designing a self-evolving run chooses:

1. `Suite`: which benchmark?
2. `AgentSurface`: what agent interface can evolve?
3. `InstanceSet`: which cases for train and heldout?
4. `surface.editable_components`: which named parts of that surface are open?
5. `strategy`: how are edits proposed?
6. `evolution.algorithm`: simple or DGM search policy.
7. `criterion`: what counts as accepted or valid?
8. `execution`: where and how runs execute.

Recipes become example configs and light wrappers. Simple and DGM share object
construction. They differ only in algorithm-specific scheduling and parent
selection.

## SWE-bench Boundary

SWE-bench owns:

- `evals/swebench/suite.py`.
- Dataset loading helpers if needed.
- Official scoring and reporting helpers.
- Container modules that know how to consume staged artifacts such as
  `AGENT_PACKAGE_KEY`.

SWE-bench does not own:

- `AgentSurface`.
- Generic rollout construction.
- Generic provider extraction.
- Generic self-evolving YAML parsing.

## Migration Plan

1. Add `InstanceSet` and compatibility with current `Slice`.
2. Add semantic `AgentSurface` and `SurfaceComponent`.
3. Add the default `python_agent_package` surface with demo components:
   `agent_program`, `prompts`, `tool_policy`, `memory_policy`, and `everything`.
4. Update `model_program_strategy(surface=..., editable_components=...)` while
   preserving `prefix=` temporarily.
5. Add `rollout_from_suite(...)`.
6. Move generic version-to-artifact packaging out of the SWE-bench evolution
   adapter and into `AgentSurface`.
7. Add `run_config.py` with typed config dataclasses and YAML loader.
8. Add `evolution/run.py` generic runner.
9. Convert simple and DGM recipes to use config-backed composition.
10. Shrink `evals/swebench/evolution_adapter.py` to SWE-bench-only helpers, or
   remove it if remaining helpers belong elsewhere.

## Testing Strategy

- Unit-test `AgentSurface.seed_files`, `validate_edits`,
  `prompt_brief`, `files_from_version`, and `artifacts_from_version`.
- Unit-test selected component path filtering for narrow components and
  `everything`.
- Unit-test `InstanceSet.sha` stability and compatibility with `Slice`.
- Unit-test `model_program_strategy(surface=...)` validation behavior.
- Unit-test `rollout_from_suite(...)` with fake backend/store and a small demo
  suite.
- Unit-test YAML loading and registry resolution without Docker or network.
- Smoke-test generic runner dry mode for simple and DGM configs.
- Keep real Docker/model execution behind `--execute`.

## Open Risks

- Renaming `Slice` may churn call sites. The implementation should use an alias
  or compatibility wrapper first, then clean names gradually.
- YAML can become a second framework if it accepts arbitrary Python imports or
  too much behavior. V1 avoids that by using registries.
- `AgentSurface` can become too path-centered again if components are not
  documented with human-readable meaning. The prompt brief should be part of the
  contract, not an afterthought.
- `AgentSurface` packaging and suite container consumption must remain loosely
  coupled. They compose through artifact keys; the surface should not import a
  benchmark suite.
- Existing SWE-bench official scoring workflows must keep parity with the
  current reporting path.

## Acceptance Criteria

- A simple self-evolving SWE-bench run can be expressed as one YAML config plus
  the generic runner.
- A DGM self-evolving SWE-bench run can reuse the same config schema and runner,
  differing mainly in the `evolution` block.
- The simple recipe no longer needs SWE-bench evolution adapter functions for
  generic rollout or surface packaging.
- `Suite`, `AgentSurface`, `SurfaceComponent`, and `InstanceSet` are each
  independently explainable in one paragraph.
- The default agent surface offers at least `everything` plus several narrower
  demo components so users do not have to reason from raw paths alone.
- No core framework object reads YAML directly.
