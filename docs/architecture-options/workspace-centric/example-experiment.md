# Example Experiment: Manager-Worker Research Team

This experiment tests whether explicit task assignment improves multi-agent research quality.

## Goal

Answer a complex research question through a manager-worker organization.

## Roles

```text
manager
  Creates tasks, assigns owners, checks completion, writes final synthesis.

researcher
  Completes evidence-gathering tasks.

critic
  Reviews artifacts and flags weak reasoning.

synthesizer
  Turns accepted artifacts into final answer.
```

## Workspace Flow

```text
manager creates task board
researchers produce artifacts
critic writes review notes
manager marks tasks accepted or needs_revision
synthesizer writes final answer
```

## Evaluation

Compare against a shared blackboard without manager assignment:

- Completion rate.
- Duplicate work.
- Final answer score.
- Coordination cost.
- Number of revisions.

