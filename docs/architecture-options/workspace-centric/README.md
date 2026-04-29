# Workspace-Centric Agents

Workspace-Centric Agents treats a multi-agent system as a shared work environment.

The central question is:

```text
How does a group of agents coordinate when they share tasks, notes, artifacts, and decisions?
```

This option is the best fit for studying organization design, role specialization, shared state, manager-worker structures, and blackboard-style collaboration.

## Runtime Shape

```text
User task
  -> Workspace initialized
  -> Agents observe workspace
  -> Agents write notes, tasks, artifacts, or decisions
  -> Coordinator or stop rule checks progress
  -> Final artifact is produced
  -> RunTraceStore + Evaluator
```

## Core Components

- `Workspace`: shared state for one run.
- `TaskBoard`: visible task items, owners, status, and dependencies.
- `Blackboard`: shared notes, claims, questions, and decisions.
- `ArtifactStore`: drafts, research notes, plans, reviews, final answers.
- `AgentSpec`: role, instruction, model, and capabilities.
- `WorkspacePolicy`: rules for who observes and writes which workspace areas.
- `Scheduler`: chooses which agents act based on task board and stop condition.
- `ContextBuilder`: builds each agent view from workspace slices.
- `RunTraceStore`: records workspace updates and agent steps.
- `Evaluator`: evaluates organization behavior and output quality.

## Minimal Data Model

```text
Workspace
  - task_board
  - blackboard
  - artifacts
  - decisions
  - run_status

TaskItem
  - id
  - title
  - owner
  - status
  - dependencies
  - result_artifact

BlackboardNote
  - id
  - author
  - kind
  - content
  - related_task

WorkspaceUpdate
  - id
  - agent
  - kind
  - target
  - content
  - step
```

## Built-In Organization Patterns

The first implementation of this option should support:

- `manager_worker`: manager creates tasks, workers complete them.
- `expert_panel`: specialists write notes, coordinator synthesizes.
- `blackboard`: all agents contribute to shared notes.
- `claim_check`: agents add claims and reviewers mark support status.
- `task_claiming`: agents choose tasks from the board.
- `coordinator_review`: coordinator accepts or sends work back.

## Context Strategy

Each agent context should be built from:

```text
role instruction
current task assignment
relevant task board items
selected blackboard notes
selected artifacts
recent workspace updates
stop condition
```

The main research control surface is workspace visibility:

- All agents see the full workspace.
- Agents see only assigned tasks.
- Specialists see their area plus global decisions.
- Coordinator sees everything.
- Reviewers see artifacts but not private notes.

## Trace and Evaluation

Workspace traces should record:

- Agent observations.
- Workspace updates.
- Task status changes.
- Artifact creation and revisions.
- Decision records.
- Final answer.
- Token usage, latency, and model calls.

Useful metrics:

- Task completion rate.
- Idle steps.
- Duplicate work.
- Rework count.
- Coordination overhead.
- Blackboard usefulness.
- Final answer score.
- Agent contribution balance.

## First Experiments

- Manager-worker vs expert panel.
- Shared blackboard vs private notes.
- Task claiming vs manager assignment.
- Coordinator review vs no coordinator.
- Full workspace visibility vs scoped visibility.

## What This Option Avoids

- No unrestricted free-form team chat as the main primitive.
- No complex workflow graph as the main primitive.
- No permission or approval layer.
- No heavy plugin system.
- No long-term memory system in the core.

