# Evolution Framework

`simple_agent_lab.evolution` is the benchmark-agnostic substrate for
self-evolving agents. It owns the durable machinery: immutable agent versions,
fair same-task comparisons, append-only decision logs, and evidence-driven
promotion. Benchmarks, Docker setup, model prompts, archive policy, and run
recipes live outside the substrate.

The short version:

```text
strategy proposes edits
        |
        v
kernel stages a candidate Version
        |
        v
rollout runs baseline and candidate on the same InstanceSet
        |
        v
reward scores each Run
        |
        v
criterion accepts or rejects
        |
        v
decision is logged, then the accepted candidate is promoted
```

## Layers

```text
src/simple_agent_lab/evolution/
  kernel/        immutable versions, current pointer, decision log, fair loop
  components/    rollout, reward, criterion, and strategy helpers
  experiment.py  small Python-facing wirer around the kernel
  surface.py     semantic editable agent surfaces
  agent_package.py  default editable Python-agent package + loader
  config.py      YAML schema and configured-run builder
  run.py         generic configured-run CLI

src/simple_agent_lab/evals/
  Suite, ContainerBackend, ArtifactStore, run_dataset, run_suite_instance

evals/swebench/
  SWE-bench host-side Suite

recipes/
  simple/        minimal config-backed self-evolving run
  dgm/           config-backed DGM-style archive recipe
```

The rule of thumb is simple: `evolution/` knows how to evolve a versioned
program; a `Suite` knows how to run benchmark instances; a recipe wires concrete
choices together.

## Core Objects

| Object | Meaning |
| --- | --- |
| `Experiment` | The direct Python entry point. It binds workspace, rollout, reward, criterion, seed files, and a train instance set. |
| `Version` | Immutable directory of agent files plus a manifest. Its directory name is the content hash. |
| `Run` | Read-only view of one executed task instance. It reads `out/result.json`, reward, and trajectory events from disk. |
| `InstanceSet` | Frozen collection of benchmark instances. `Slice` is the compatibility name still used in older evolution APIs and logs. |
| `Proposal` | A strategy's candidate edit: full file contents or `None` tombstones, plus note/evidence/base/kind. |
| `Context` | What a strategy sees: current version, previous decisions, baseline runs, reward function, and helpers. |
| `Verdict` | A criterion's accept/reject result plus reason and deltas. |
| `Decision` | Append-only record of the baseline/candidate comparison. |
| `AgentSurface` | The editable shape of an agent: seed files, semantic components, validators, and run artifacts. |

The kernel stores everything on disk so a run can be inspected without loading
framework state into memory:

```text
<run-root>/evolution/
  versions/<hash>/...
  pointers/current.json
  decisions.jsonl

<run-root>/runs/
  <version-hash>-<instance-set-sha>/<instance-id>/out/result.json
```

Configured simple runs also write heldout reports under:

```text
<run-root>/evaluation/summary.json
```

## Direct Python Use

Use direct Python when you are building a custom experiment or a new recipe and
want the smallest surface area.

```python
from simple_agent_lab.evolution import Experiment, Proposal
from simple_agent_lab.evolution.components.criterion import improve
from simple_agent_lab.evolution.components.reward import result_key


def my_strategy(ctx):
    if not ctx.failures:
        return None
    return Proposal(
        edits={"agent/prompt.md": "Try a more direct repair plan.\n"},
        note="tighten repair prompt",
    )


exp = Experiment(
    "evals/out/my-evolution/evolution",
    rollout=my_rollout,              # (Version, InstanceSet) -> Sequence[Run]
    reward=result_key,               # Run -> float | {dim: float}
    criterion=improve("reward"),     # compare aggregate scores
    slice_id="train",
    instances=train_records,
    seed={"agent/prompt.md": "You fix bugs.\n"},
)

decision = exp.step(my_strategy)
print(exp.history())
```

The substrate does not care whether `my_rollout` runs Docker, an in-process
toy task, a remote worker pool, or a non-code benchmark. It only needs `Run`
objects whose `out/result.json` can be scored.

## Configured Simple Runs

Use the YAML path when you want a small runnable recipe. The fastest entry point
is [`recipes/simple/README.md`](../../../recipes/simple/README.md), which shows
the dry-run, real run, train-only smoke, and artifact map. At the framework
level, a config names concrete factories; recipe code registers those names
before calling `simple_agent_lab.evolution.run`.

```yaml
run:
  id: simple-swebench-demo
  output_root: evals/out/self_evolving/simple
  execute: false
  reset: false
  dotenv: .env

suite:
  name: swebench
  args:
    in_env_scoring: true

surface:
  name: source_tree
  editable_components: [everything]
  artifact_key: source_tree
  default: current_source_tree

instances:
  train:
    id: train
    path: configs/swebench/demo-train-60.jsonl
  heldout:
    id: heldout
    path: configs/swebench/demo-test-60.jsonl

execution:
  backend:
    name: local_docker
  store:
    name: local_dir
  parallel: 1
  max_turns: 75

model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
  base_url_env: OPENAI_BASE_URL

strategy:
  name: source_tree_agent
  args:
    repo_root: .
    system_prompt: |
      You are a meta-agent evolving a SWE-bench coding agent.

      Inspect and edit files under src/simple_agent_lab/ in the temporary
      repository copy. Run cheap checks when practical, then finish with a short
      summary.

evolution:
  algorithm: simple
  rounds: 4
  criterion:
    name: promote_not_worse
    args:
      dim: reward

evaluation:
  baseline_heldout: true
  final_heldout: true
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: false
```

Run it through the simple recipe:

```bash
bash runs/run_self_evolving_simple.sh --run-id simple-smoke
bash runs/run_self_evolving_simple.sh --run-id simple-real --execute
```

Dry-runs print the resolved plan. Executed simple runs train on
`instances.train`, optionally evaluate `instances.heldout` before/final or every
N rounds, and write `evaluation/summary.json` with reward means, resolved counts
when the suite reports them, and before/final deltas.

For a cheap train-only smoke, copy `configs/simple_swebench.yaml`, point
`instances.train.path` at a tiny JSONL split, set `instances.heldout: null`,
disable `evaluation.baseline_heldout` and `evaluation.final_heldout`, set
`evolution.rounds` to `3` or `4`, and set `execution.parallel` to the number of
train instances to run concurrently.

## Agent Surfaces

`AgentSurface` makes the editable target semantic instead of just "whatever path
the model writes." A surface defines:

- default seed files for the initial `Version`
- named editable components such as `agent_program`, `prompts`, `tool_policy`,
  `memory_policy`, or `everything`
- validators such as path safety, Python syntax, and required entrypoint checks
- how a `Version` becomes per-run input artifacts

The built-in `source_tree_agent_surface(...)` seeds Python files from
`src/simple_agent_lab/` into the initial `Version`. Config-built source-tree
runs stage the selected version under `input/source_tree/src/simple_agent_lab/`
and prepend `/agent/run/input/source_tree/src` to the candidate run's
`PYTHONPATH`, so the suite exercises the evolved framework source.

The lower-level `python_agent_surface(...)` remains available for tests and
custom wrapper-package experiments. It stores files under `agent/`, requires
`agent/agent_program.py:build_agent`, and packages the selected version files as
`input/agent_package.json` for eval containers that opt into that artifact.

## Benchmark Execution

The evolution framework does not introduce a benchmark adapter layer of its own.
It reuses the eval framework:

- a host-side `Suite` maps records to launch specs and task inputs
- a `ContainerBackend` runs the suite in Docker, locally, or remotely
- an `ArtifactStore` moves inputs and outputs
- `run_dataset(...)` executes an `InstanceSet`
- `rollout_from_suite(...)` adapts that dataset run into `(Version, InstanceSet)
  -> Sequence[Run]`

For SWE-bench, the normal benchmark interface is `evals/swebench/suite.py`
`SwebenchSuite`. The simple recipe registers it under the name `swebench` and
points the suite at the in-wheel `simple_agent_lab.evals.suites.swebench.evolving`
container hook. Source-tree runs stage the evolved framework source as a
candidate source tree; DGM adds recipe-local helpers for archive-specific
official scoring.

## Strategies, Rewards, and Criteria

Strategies return `Proposal | None`.

- `source_tree_agent_strategy(...)` lets a bash-capable meta-agent edit a
  temporary repo copy after reading `SELF_EVOLUTION_CONTEXT.md`, then derives a
  `Proposal` from filesystem diffs under `src/simple_agent_lab/`.
- `recipes.dgm.evolve.dgm_agentic_strategy(...)` is recipe-local DGM policy: it
  materializes the selected parent `agent/` package, loads that package as a SAL
  agent, lets it perform its own self-improvement step, and converts `agent/`
  diffs into a `Proposal`.
- `recipes.ahe.strategy.ahe_agent_strategy(...)` is recipe-local AHE policy: it
  runs the analyzer, stages the analysis into a harness workspace, lets a
  role-separated SAL evolve agent edit `harness/`, and records
  `change_manifest.json`.
- `model_program_strategy(...)` is a lower-level wrapper-program strategy that
  asks an LLM to rewrite full files under the selected surface. It validates
  JSON, filters invalid edits, and returns a `Proposal`.
- Custom strategies can be ordinary Python functions. This is useful for tests,
  ablations, or hand-written evolution steps.

Rewards turn one `Run` into either a float or named dimensions.

- `result_key(run)` reads the `reward` key from `out/result.json`.
- `cost_tokens(run)` sums token usage from trajectory events.
- Custom rewards can return mappings such as `{"reward": 0.7, "cost": 1200}`.

Criteria compare aggregated baseline and candidate scores.

- `improve(dim)` accepts strict improvement.
- `promote_not_worse(dim, tol=...)` accepts candidates that do not regress past
  a tolerance.
- `valid_when(dim)` admits any gradable child; DGM uses this for archive
  admission.
- `guarded(objective, guards)` optimizes one metric subject to guard metrics.

## Simple vs DGM

Both recipes share the same substrate guarantees.

| Recipe | Purpose | Loop policy | Performance report |
| --- | --- | --- | --- |
| `recipes/simple/` | Minimal framework use | Sequential current-version evolution | Generic suite-scored heldout `evaluation/summary.json` |
| `recipes/dgm/` | DGM-style archive/open-ended mechanics with parent-agent self-improvement | Parallel branches, open-ended archive admission, parent selection | Archive-specific official before/final artifacts and `test_summary.json` |

DGM's parent-agent self-improvement, archive selection, branch scheduling, and
official heldout workflow are recipe policy. Its YAML schema lives in
`recipes/dgm/config.py` and `configs/dgm_swebench.yaml`, not in the generic
`evolution.config` builder, so the user experience stays consistent without
promoting DGM semantics into the substrate. Describe it as a compact DGM-style
teaching recipe unless a run report also establishes the broader research
claims.

## Writing a New Recipe

To add a benchmark or experiment shape:

1. Implement or reuse a `Suite`.
2. Choose an `AgentSurface`.
3. Build a rollout, usually with `rollout_from_suite(...)`.
4. Pick a reward and criterion.
5. Write or configure a strategy.
6. Feed train instances into `Experiment`.
7. Keep benchmark setup and host/Docker probing in `recipes/`, not in
   `src/simple_agent_lab/evolution/`.

For YAML-backed recipes, register factories before loading config:

```python
from simple_agent_lab.evolution import registry

registry.SUITES["my_suite"] = lambda **args: MySuite(**args)
registry.SURFACES["my_surface"] = lambda **args: my_surface(**args)
registry.BACKENDS["my_backend"] = lambda **args: MyBackend(**args)
registry.STORES["my_store"] = lambda root, **args: MyStore(root, **args)
registry.STRATEGIES["my_strategy"] = my_strategy_factory
```

Then the config can refer to `suite.name: my_suite`,
`surface.name: my_surface`, and so on.

## Boundaries

Keep these boundaries intact:

- `src/simple_agent_lab/evolution/` must stay benchmark-agnostic.
- The substrate should not import host-side `evals/` adapters such as
  `evals/swebench`.
- Docker and environment probing belong in recipes or eval backends, not in the
  evolution kernel.
- DGM archive policy belongs under `recipes/dgm/algorithm/`.
- Benchmark scoring should flow through the suite/run result contract or
  recipe-local reporting, not through benchmark-specific substrate code.

The durable architectural rationale lives in
[`docs/decisions/20260617-recipes-as-the-self-evolving-surface.md`](../../../docs/decisions/20260617-recipes-as-the-self-evolving-surface.md).
The agent-facing loading guide is
[`docs/agent-native/self-evolving.md`](../../../docs/agent-native/self-evolving.md).
