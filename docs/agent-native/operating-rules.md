# Operating Rules

Read when:

- You are changing runtime behavior, message contracts, context visibility,
  tool execution, eval/trajectory records, validation gates, or repo docs.

Do not read for:

- Small typo fixes in docs that do not change loading paths or source-of-truth
  rules.
- Purely mechanical formatting changes that do not alter behavior.

Sources:

- Working tree inspected on 2026-05-11.
- `AGENTS.md`, `README.md`, `CONTEXT.md`, `docs/agent-native/`, `docs/decisions/`,
  `src/simple_agent_lab/`, `runs/README.md`, `tests/README.md`,
  `evals/README.md`, and `evals/swebench/README.md`.

Freshness:

- Re-check `git status`, `README.md`, `docs/decisions/README.md`, and
  `runs/README.md` before relying on this doc after runtime or eval changes.
- Live external provider and benchmark setup is not verified by this doc.

## Source-Of-Truth Map

| Disputed fact | Source that wins | Agent action |
| --- | --- | --- |
| What runtime is canonical | `src/simple_agent_lab/core.py`, ADR 0009 | Do not revive retired design-version copies as implementation targets. |
| Whether a new abstraction belongs in core | ADR 0001, ADR 0009, `docs/agent-native/code-style.md` | Keep additions small and explicit unless an ADR accepts the tradeoff. |
| Message and provider-boundary shape | `CONTEXT.md`, `src/simple_agent_lab/messages.py`, ADR 0006, `src/simple_agent_lab/llm/README.md` | Keep runtime routing fields out of provider-boundary payloads. |
| Context visibility and budget behavior | `src/simple_agent_lab/context_view.py`, ADR 0010, `tests/unit/test_core.py`, `tests/unit/test_token_usage.py` | Preserve full history in `State`; project model-visible context explicitly. |
| Tool result semantics | `src/simple_agent_lab/tools/`, `src/simple_agent_lab/core.py`, bash tests | Tool outputs become `tool_result` messages; `details` are local inspection data. |
| Trace/eval/training separation | `src/simple_agent_lab/trajectory.py`, `evaluation.py`, `training_data.py`, ADR 0008 | Do not put scores or training labels into raw trajectory records. |
| Benchmark suite boundaries | ADR 0011, `evals/swebench/README.md` | Keep suite-specific heavy dependencies outside the minimal core runtime. |
| Local quality gate | `docs/agent-native/development.md`, `runs/run_ci.sh`, `.github/workflows/ci.yml` | Keep local and remote gates in lockstep when checks change. |
| Architecture decisions | `docs/decisions/` | Use this repo's ADR directory; do not create `docs/adr/`. |
| Normal repo boundary | Owner confirmation on 2026-05-11, `docs/agent-native/README.md` | Treat this checkout as self-contained unless a task explicitly names an external repo or system. |
| First live provider adapter target | Owner confirmation on 2026-05-11, `src/simple_agent_lab/llm/README.md` | Implement `openai-chat` first; keep live-provider smoke opt-in and outside required CI unless owner policy changes. |

## Directly Editable

Agents can usually edit these with code evidence and a narrow check:

- Small runtime behavior covered by focused unit tests.
- New deterministic examples that keep defaults runnable without external
  services.
- Documentation routing, stale links, and inventory freshness.
- Reference architecture notes before implementation begins.
- Task specs for handoffable work.

## Needs More Evidence

Collect code, doc, or config evidence before editing:

- Public exports in `src/simple_agent_lab/__init__.py`.
- Message role names, content block shapes, tool-call fields, and token usage.
- Vocabulary that affects how docs distinguish `Message`, `ModelMessage`,
  provider payloads, and content blocks.
- Context budget rules, clipping, or grouping around tool-call/tool-result
  pairs.
- CI gate changes in `runs/run_ci.sh`, `.github/workflows/ci.yml`, and
  `docs/agent-native/development.md`.
- Eval output schema changes under `evals/out/` or shared record modules.

## Needs Owner Confirmation

Ask or record a question before changing:

- Changing the first live provider adapter away from `openai-chat`.
- Whether external benchmark dependencies such as SWE-bench and Docker should
  become part of required CI or remain optional smoke paths.
- Release ownership, reviewer expectations, and publishing flow after the first
  open-source release prep.

## Validation Map

| Change type | Narrow check | Broader check |
| --- | --- | --- |
| Formatting-only or broad Python edits | `uv run ruff format --check .` | `bash runs/run_ci.sh` |
| Core runtime, messages, tools, context | `uv run python -m unittest discover -s tests/unit` | `bash runs/run_ci.sh` |
| Type or public API surface | `uv run ty check src` | `bash runs/run_ci.sh` |
| Public example behavior | `bash runs/run_examples.sh` | `bash runs/run_ci.sh` if package code changed |
| Bash-use agent demo | `bash runs/run_bash_agent_demo.sh` and `uv run python -m unittest tests.test_bash_agent` | `bash runs/run_ci.sh` |
| SWE-bench adapter plumbing | `bash runs/run_swebench_smoke.sh` | `bash runs/run_swebench_gold_smoke.sh` only when optional SWE-bench and Docker setup exist |
| Docs-only routing change | Review links and loading map manually; run the smallest affected command only if examples or commands changed | `bash runs/run_ci.sh` is optional unless code or command contracts changed |

## Hidden Boundaries Worth Preserving

- The project optimizes for students and small teams. Beginner readability is a
  product constraint, not only a style preference.
- `State.events` is the inspectable run trace. Do not add a separate trace store
  until a runnable experiment proves the need.
- `context_view` is a projection, not storage. It may trim model-visible input,
  but it must not mutate durable history.
- `TokenUsage.input_tokens` is per-call aggregate data. Do not split it across
  earlier input messages as if it were per-message accounting.
- The fake LLM adapter is a deterministic boundary exerciser. Examples should
  not bypass the real message/tool path by stuffing response text into
  provider-specific `extra`.
- SWE-bench gold `patch` and `test_patch` fields belong to scoring, not the
  model-visible task.
- `evals/out/` is generated local artifact data and should not become source of
  truth.

## ADR Boundary

Create or update an ADR under `docs/decisions/` only when all of these are true:

- The decision is hard or expensive to reverse.
- Future agents or engineers would otherwise wonder why the system is shaped
  this way.
- There is a real tradeoff between credible alternatives.

Otherwise, update the relevant context doc, task spec, reference note, test, or
run script.
