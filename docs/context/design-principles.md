# Design Principles

## Clarity First

The first implementation should be understandable by reading the code directly. Prefer explicit function calls and plain data structures.

## Hackable by Default

Users should be able to change prompts, tools, model adapters, and orchestration logic without learning a large internal framework.

## Small Core, Visible Edges

The core agent loop should stay small. Integrations should sit at visible edges so they can be replaced or removed.

## Learn Before Abstracting

Add abstractions only after repeated patterns are clear. Early code should teach the shape of the system.

## Agent-Friendly Context

The repository should make it easy for coding agents to understand intent, constraints, and next steps before editing files.

