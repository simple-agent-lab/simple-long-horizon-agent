---
title: "Use Harbor as the Eval Harness for Harbor Datasets"
status: Accepted
date: 2026-07-06
slug: harbor-as-eval-harness
---

# Use Harbor as the Eval Harness for Harbor Datasets

## Status

Accepted

## Context

Simple Agent Lab already has a generic containerized eval framework for suites
that SAL owns directly, such as SWE-bench and ProgramBench. Harbor is a
different kind of integration: it is itself a benchmark harness with dataset
resolution, task environments, verifier execution, artifact collection, and job
aggregation.

The goal is to evaluate SAL agents on Harbor-supported code and tool-use
datasets without adding one SAL suite per Harbor dataset and without
reimplementing Harbor's dataset registry. Harbor also has a custom installed
agent interface, so SAL can run inside the task environment rather than driving
tools from the host.

The main architectural risk is accidentally creating a second tool-execution
transport: a host-side SAL agent that forwards every bash/read/tool call into
the Harbor container. That would duplicate Harbor environment semantics,
complicate traces, and make failures harder to explain.

## Decision

Add a single `harbor` bench entry that shells out to `harbor run` and passes SAL
as a Harbor custom installed-agent import path:

```text
simple_agent_lab.evals.harbor.agent:SimpleAgentLabHarborAgent
```

The Harbor adapter installs a small SAL runner inside the Harbor task
environment, uploads the task instruction to `/logs/agent`, and starts the
runner with one Harbor `environment.exec` call. When the adapter is imported
from a local SAL checkout, it uploads a minimal source archive and installs that
source into the task container; otherwise the configurable `sal_package` value
is installed. After that, the normal SAL agent loop runs inside the same
environment and its bash/read/task tools execute locally in that container.
Harbor remains responsible for task setup, verification, artifact download, and
`result.json` aggregation.

There is no `harbor_exec` tool, no host-side per-command forwarding, and no
fallback path where a host agent sends individual tool commands into the
container.

Harbor is optional. SAL core modules and required tests must import without
Harbor installed. The `harbor` optional dependency is available only for Python
versions that satisfy Harbor's requirements.

## Consequences

- **Easier.** One SAL bench unlocks Harbor datasets that fit SAL's current
  code/tool-use agent shape. Future dataset additions happen in Harbor, not by
  adding SAL suite modules.
- **Clear ownership.** Harbor owns dataset/task/environment/verifier/result
  behavior; SAL owns only agent runtime behavior and SAL trace files.
- **Container-local tools.** The same SAL bash/read/task implementations used by
  other evals run inside the Harbor task container, so tool behavior is visible
  by inspecting SAL code and the Harbor trial logs.
- **Optional dependency boundary.** Importing `simple_agent_lab` or running core
  unit tests does not require Harbor. Harbor-specific code lazily imports Harbor
  only when the installed-agent class is constructed by Harbor.
- **Operational dependency.** Real Harbor runs need a Python 3.12+ environment
  with Harbor installed and a Harbor-compatible task environment. The required
  unit path uses `--dry-run` and the fake provider instead.
- **Out of scope.** SAL does not implement Harbor dataset discovery, Harbor
  verification, Harbor result aggregation, or Harbor's trajectory format. SAL
  writes its own trace JSONL under the agent logs for debugging.

## Alternatives Considered

- **Add one SAL suite per Harbor dataset.** This duplicates Harbor's registry and
  environment work, increases maintenance, and makes every new Harbor dataset a
  SAL code change.
- **Run SAL on the host and forward tools into Harbor with a `harbor_exec`
  fallback.** Rejected. It creates two tool execution semantics, makes each SAL
  tool call depend on Harbor transport behavior, and is unnecessary because
  Harbor installed agents can run inside the task environment.
- **Depend on Harbor from core SAL imports.** Rejected. Harbor is useful for an
  eval path, not for the teaching runtime, and currently has a narrower Python
  version floor than SAL.
