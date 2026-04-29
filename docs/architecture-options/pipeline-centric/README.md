# Pipeline-Centric Agents

Pipeline-Centric Agents treats a multi-agent system as a reproducible task flow.

The central question is:

```text
How should work be decomposed, handed off, merged, reviewed, and measured?
```

This option is the best fit for staged research workflows, benchmarkable experiments, map-reduce style multi-agent tasks, and clear handoff contracts.

## Runtime Shape

```text
User task
  -> PipelineSpec
  -> Scheduler finds ready nodes
  -> AgentRunner executes node agents
  -> ArtifactStore records outputs
  -> downstream nodes receive artifacts
  -> final node produces answer
  -> RunTraceStore + Evaluator
```

## Core Components

- `PipelineSpec`: nodes, edges, entrypoint, and stop condition.
- `NodeSpec`: a stage in the workflow, usually backed by one agent.
- `EdgeSpec`: a handoff from one node artifact to another node input.
- `AgentSpec`: role and instruction for a node agent.
- `AgentRunner`: executes a node using the shared `AgentStep` concept.
- `ArtifactStore`: stores plans, notes, drafts, reviews, and final answers.
- `Scheduler`: runs ready nodes, including parallel fan-out and fan-in.
- `ContextBuilder`: builds node context from upstream artifacts.
- `RunTraceStore`: records node execution, artifacts, model calls, and metrics.
- `Evaluator`: compares pipeline variants and output quality.

## Minimal Data Model

```text
PipelineSpec
  - id
  - nodes
  - edges
  - entry
  - final
  - max_steps

NodeSpec
  - id
  - agent
  - input_artifacts
  - output_artifact
  - run_when

EdgeSpec
  - from_node
  - from_artifact
  - to_node
  - to_input

Artifact
  - id
  - type
  - producer
  - content
  - metadata
```

## Built-In Pipeline Patterns

The first implementation of this option should support:

- `planner_executor`: plan first, execute second.
- `research_map_reduce`: parallel researchers, then synthesis.
- `writer_reviewer`: draft, review, revise.
- `self_refine`: same agent or role loops over draft and critique.
- `compare_then_select`: multiple independent answers, then selector.
- `retrieve_synthesize_verify`: gather evidence, write answer, verify claims.

## Context Strategy

Each node context should be built from:

```text
node instruction
task objective
selected upstream artifacts
pipeline metadata
artifact output contract
```

The main research control surface is the handoff contract:

- Full upstream artifact.
- Summary only.
- Structured fields only.
- Multiple upstream artifacts merged.
- Reviewer sees draft but not original notes.

## Trace and Evaluation

Pipeline traces should record:

- Node start and finish events.
- Inputs each node received.
- Artifacts each node produced.
- Parallel execution groups.
- Revision loops.
- Final answer.
- Token usage, latency, and model calls per node.

Useful metrics:

- Final answer score.
- Per-stage cost.
- Per-stage latency.
- Artifact quality.
- Handoff loss.
- Revision improvement.
- Pipeline success rate.
- Parallel speedup.

## First Experiments

- Single agent vs planner-executor.
- One researcher vs three parallel researchers.
- Writer-only vs writer-reviewer-reviser.
- Full handoff vs structured handoff.
- Synthesis with and without verifier.

## What This Option Avoids

- No free-form message bus as the primary model.
- No complex organization or team lifecycle.
- No permission or approval layer.
- No large tool discovery system.
- No long-term memory as a core dependency.

