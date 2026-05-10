# Reference Architecture: mini-SWE-agent

## Source

- GitHub: `https://github.com/SWE-agent/mini-swe-agent/blob/v2.0/src/minisweagent/agents/default.py`
- Date reviewed: 2026-05-08

## Summary

mini-SWE-agent is useful for Simple Agent Lab because its default agent keeps
the control flow small and visible:

```text
run(task)
  -> add system and task messages
  -> step()
     -> query model
     -> execute actions in environment
     -> append observations
  -> repeat until exit
  -> save trajectory
```

The important split is `Agent` owns the loop, `Model` owns the next assistant
message, and `Environment` owns bash execution. The transcript is the shared
state between those pieces.

## What To Borrow

- Keep the readable loop shape: query, execute, observe, repeat.
- Treat bash execution as an environment/tool boundary, not as hidden logic in
  the model adapter.
- Keep the trajectory as ordinary messages and events so a learner can inspect
  what happened after the run.
- Include small limits around steps, cost, timeout, or output size before the
  demo can hang or flood context.

## What To Avoid For This Repo

- Do not copy the full configuration stack before the teaching demo needs it.
- Do not introduce a second runtime class when `AgentRuntime` already provides
  the loop, trace, scheduler, and tool-result path.
- Do not make bash a privileged production abstraction. The first version is a
  deterministic local demo tool, not a sandbox or permission system.

## Current Simple Agent Lab Mapping

The local bash-use demo maps the mini-SWE shape onto the promoted runtime:

| mini-SWE shape | Simple Agent Lab shape |
| --- | --- |
| `DefaultAgent.run()` | `AgentRuntime.prompt(...)` |
| `step()` | one scheduled `Agent.step(...)` turn |
| `model.query(messages)` | `make_llm_step(...)` through the LLM layer |
| `env.execute(action)` | `make_bash_tool(...).execute(...)` |
| observation messages | `ToolResultMessage` in `State.events` |
| trajectory save | printable `State.events` trace |

The demo intentionally uses the fake LLM adapter so the example remains
deterministic while still exercising the real model/tool/message boundary.
