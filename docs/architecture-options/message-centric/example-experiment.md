# Example Experiment: Debate With Judge

This experiment compares a debate protocol against a single-agent baseline.

## Goal

Answer a research question with better factual coverage and reasoning quality than a single agent.

## Agents

```text
proposer
  Role: produce an initial answer.

critic
  Role: identify missing assumptions, weak evidence, and contradictions.

defender
  Role: revise or defend the answer after critique.

judge
  Role: select the final answer and explain the decision.
```

## Message Flow

```text
round 0: user -> proposer
round 1: proposer -> broadcast
round 2: critic -> broadcast
round 3: defender -> judge
round 4: judge -> final
```

## Evaluation

Compare against a single agent on:

- Final answer quality.
- Factual coverage.
- Number of unsupported claims.
- Token cost.
- Time to final answer.

