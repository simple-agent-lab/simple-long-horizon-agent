---
title: "Treat Self-Evolution As Harness Capability"
status: Accepted
date: 2026-05-07
slug: treat-self-evolution-as-harness-capability
---

# Treat Self-Evolution As Harness Capability

## Status

Accepted

## Context

Self-evolving agents are becoming a serious research direction. The survey
"A Survey of Self-Evolving Agents" organizes the field around four practical
questions:

- What evolves: model, memory, prompt, tools, or architecture.
- When evolution happens: inside one run or across runs.
- How evolution is guided: feedback, rewards, demonstrations, or population
  search.
- Where evolution is evaluated: general tasks, specialized domains, and
  long-horizon learning settings.

Simple Agent Lab should learn from that framing without turning into a heavy
training framework or a benchmark platform. The project mission is still to
make agent systems easy to understand, modify, and teach.

The risk is that "self-evolution" becomes magic: an agent rewrites prompts,
tools, memory, or code without a clear trace, comparison, or rollback point.
That would conflict with the harness engineering workflow and make the system
harder for students and future agents to inspect.

## Decision

Treat self-evolution as a harness capability, not as hidden runtime behavior.

The first self-evolution loop should be explicit:

```text
run -> trace -> evaluate -> propose candidate -> compare -> accept or reject
```

The default evolution targets are non-parametric and beginner-readable:

- prompt text
- context visibility rules
- memory summaries or lessons
- tool descriptions
- small recipes built on the runtime

Model-weight training, arbitrary self-modification of the core runtime, and
open-ended architecture search are out of scope for the first implementation.
They may be studied later as reference architectures or advanced examples, but
they should not become the default path.

Every accepted evolution must leave a local record:

- the task or eval set used as the feedback signal
- the baseline behavior
- the candidate change
- the comparison result
- the reason the candidate was accepted or rejected

Runtime designs should therefore be compared partly by how easy they make this
loop to observe and test. A useful self-evolution base is not the one with the
most automation. It is the one where another person or agent can inspect why a
change happened and whether it helped.

## Consequences

Self-evolution reinforces the existing harness engineering direction. Trace,
evaluation, and versioned feedback become core teaching concepts instead of
optional tooling.

The project can experiment with self-improvement without claiming to be a
general self-improving AI system. Early examples should optimize small,
observable behavior under narrow checks.

The canonical runtime should expose enough state for evals to compare runs.
If a runtime hides context construction, tool results, model payloads, or
acceptance decisions, it is a weaker fit for this direction.

The tradeoff is that the first self-evolution system will be conservative. It
will not compete directly with systems that evolve agent code, train policies,
or search large workflow graphs. That is acceptable because Simple Agent Lab is
optimized for legibility before scale.

## Alternatives Considered

- Make self-evolution the core runtime abstraction. Rejected because it would
  force every beginner example to understand eval loops, candidate selection,
  and versioning before understanding a basic agent loop.
- Build a training-first self-evolving agent framework. Rejected because it
  would add provider, environment, dataset, and compute complexity before the
  educational runtime is settled.
- Allow agents to freely rewrite the repository. Rejected because it weakens
  traceability and makes feedback signals hard to trust.
- Treat self-evolution as only a reference-architecture note. Rejected because
  the idea should influence runtime selection and future harness design.
