# Components

## PipelineSpec

Defines the workflow graph.

The graph should be simple enough to inspect directly. A first implementation only needs DAG-style pipelines plus bounded revision loops.

## NodeSpec

Defines one stage.

Each node should have an explicit input contract and output artifact type.

## ArtifactStore

Stores node outputs.

Artifacts are the main handoff mechanism. They should be stable, inspectable, and referenced by ID.

## Scheduler

Runs nodes when their required inputs exist.

The first scheduler should support:

- Sequential nodes.
- Parallel fan-out.
- Fan-in after all required artifacts exist.
- Bounded loops for revise-and-review.

## ContextBuilder

Builds node context from upstream artifacts.

This makes handoff design explicit and testable.

## RunTraceStore and Evaluator

Trace facts by node and artifact.

Evaluator reports should make it easy to compare two pipeline specs on the same task.

