# Existing Documentation Inventory

Read when:

- You need to know whether repo knowledge already exists before adding new docs.
- You are changing doc roles, freshness, loading triggers, or the agent-native
  entrypoint.

Sources:

- Inventory refreshed from the working tree on 2026-05-11.
- `rg --hidden --files -g '*.md'`
- Focused reads of `README.md`, `CONTEXT.md`, `CONTRIBUTING.md`,
  `CHANGELOG.md`, `.agents/skills/docs-sync/SKILL.md`, `.github` templates,
  `docs/agent-native/`, `docs/decisions/`, `runs/README.md`, `tests/README.md`,
  `evals/README.md`, and `evals/swebench/README.md`.

## Canonical Docs

| Doc | Role in system | Canonical status | Read when / loaded by | Freshness | Action |
| --- | --- | --- | --- | --- | --- |
| `AGENTS.md` | First-hop collaboration contract | Canonical | Every agent session | Current | Keep short; route to agent-native docs, do not duplicate topic detail. |
| `README.md` | Public project map and current status | Canonical | Repo tour, setup, current status | Current | Link new doc roots, but keep public-facing. |
| `CONTEXT.md` | Root vocabulary and resolved terminology boundaries | Canonical support doc | Message/protocol/provider terminology work | Current | Load with ADR 0006 and message code. |
| `CONTRIBUTING.md` | Human contributor guide | Canonical support doc | PR and contribution workflow | Current | Link only. |
| `CHANGELOG.md` | Release history | Canonical support doc | Release notes or package history | Current for 0.1.0 | Update on releases, not routine code changes. |
| `docs/agent-native/README.md` | Agent-native entrypoint and loading map | Canonical | Future agent maintenance or ambiguous tasks | Current | Update when loading triggers change. |
| `docs/agent-native/doc-inventory.md` | Existing-doc roles, freshness, overlaps, and gaps | Canonical | Before adding or reorganizing docs | Current | Update with doc role/freshness changes. |
| `docs/agent-native/operating-rules.md` | Source-of-truth and validation routing | Canonical | Runtime, protocol, eval, or validation changes | Current | Keep rules here; link rather than duplicate from README. |
| `docs/agent-native/owner-questions.md` | Unresolved owner/system questions | Canonical queue | When an answer cannot be proven from code/docs | Current | Move confirmed answers into canonical docs. |
| `docs/agent-native/project-intent.md` | Product intent, target users, and design principles | Canonical topic doc | Product direction, teaching audience, or taste changes | Current; merged from former `docs/context/*` docs on 2026-05-11 | Link only. |
| `docs/agent-native/code-style.md` | Code style and repository shape | Canonical topic doc | Python/source layout/style changes | Current; moved from `docs/context/` on 2026-05-11 | Link only. |
| `docs/agent-native/harness-engineering.md` | Harness workflow | Canonical topic doc | Non-trivial implementation or docs process | Current; moved from `docs/context/` on 2026-05-11 | Link only. |
| `docs/agent-native/development.md` | Local/remote quality gate | Canonical topic doc | Commands, CI, dev dependencies | Current; moved from `docs/context/` on 2026-05-11 | Link only. |
| `docs/decisions/README.md` | ADR index | Canonical | Architecture decision lookup | Updated 2026-05-11 | Keep accepted ADR list complete. |
| `docs/decisions/0000-template.md` | ADR template | Canonical template | Creating a new ADR | Current | Link only. |
| `docs/decisions/0001-0011*.md` | Accepted ADRs | Canonical historical decisions | Architecture or hard-to-reverse changes | Current | Link from agent-native loading map. |
| `docs/reference-architectures/README.md` | Reference architecture index | Canonical | Borrowing from external systems | Current | Link only. |
| `docs/reference-architectures/*.md` | External architecture notes | Supporting references | Specific external design comparison | Mixed | Use as evidence/rationale, not current runtime authority. |
| `docs/glossary.md` | Shared vocabulary | Supporting | Term clarification | Current | Link only. |
| `docs/README.md` | Human-facing docs navigation | Canonical support doc | Browsing the doc tree from the repo root | Added 2026-05-11 | Keep in sync with the three doc roots and `glossary.md`. |
| `runs/README.md` | Runnable command index | Canonical validation map | Smoke runs and local checks | Current | Link only. |
| `tests/README.md` | Unit-test scope | Canonical validation map | Behavioral test strategy | Current | Link only. |
| `evals/README.md` | Eval layer overview | Canonical validation map | Eval/training comparison work | Current | Link only. |
| `evals/swebench/README.md` | SWE-bench adapter guide | Canonical suite runbook | SWE-bench adapter work | Current | Link only. |
| `src/simple_agent_lab/llm/README.md` | LLM boundary guide | Canonical package doc | Provider adapter or bridge work | Current, with TODO adapter roadmap | Link only; do not treat TODO providers as implemented. |
| `.github/pull_request_template.md` | PR checklist | Canonical support doc | PR preparation | Current | Keep aligned with local quality gate and ADR expectations. |
| `.github/ISSUE_TEMPLATE/*.md` | Issue intake templates | Canonical support docs | Bug or feature-report process | Current | Link only. |
| `.agents/skills/docs-sync/SKILL.md` | Repo-local docs-sync helper skill | Supporting agent workflow | User asks to sync docs with code | Updated 2026-05-11 | Keep current with repo paths and run scripts. |

## Historical Or Supporting Docs

| Doc | Role | Canonical source today | Action |
| --- | --- | --- | --- |
| Long reference notes under `docs/reference-architectures/` | Generated or researched external references | Accepted ADRs plus current `src/` for committed behavior | Keep; load only when the task names that reference. |

The earlier `docs/architecture-options/` tree (three pre-consolidation
runtime write-ups) was removed on 2026-05-11. Use ADRs 0001, 0005, and
0009 plus `src/simple_agent_lab/core.py` for the accepted direction. The
former `docs/tasks/` task-spec tree was removed the same day; handoff
context now lives in conversation, ADRs, or the relevant agent-native doc.

## Duplicates And Overlaps

| Docs | Overlap | Canonical source | Action |
| --- | --- | --- | --- |
| `README.md`, ADR 0001, ADR 0009 | Runtime shape and history | `src/simple_agent_lab/core.py` for behavior; ADRs for decisions | Agent-native docs route to these; avoid restating the implementation. |
| `CONTEXT.md`, `docs/glossary.md`, ADR 0006, message code | Vocabulary and protocol semantics | `CONTEXT.md` for terminology; code for behavior | Load both for message/protocol changes. |
| `docs/agent-native/development.md`, `runs/README.md`, `.github/workflows/ci.yml` | Local and remote checks | `docs/agent-native/development.md` for policy; `runs/run_ci.sh` and workflow for commands | Keep in lockstep when checks change. |
| ADR 0008, ADR 0011, `evals/README.md`, `evals/swebench/README.md` | Eval and trajectory split | ADRs for decisions; suite README for command shape | Keep suite details outside core runtime docs. |
| `docs/agent-native/README.md` and this inventory | Loading map vs doc roles | README routes by task; inventory records doc roles/freshness | Keep task routing in the README; keep role/freshness/gap maintenance here. |

## Loading Role Boundaries

- Task-to-doc routing lives in `docs/agent-native/README.md`.
- Existing-doc status, overlaps, stale areas, and missing knowledge live here.
- Stop conditions and validation rules live in `docs/agent-native/operating-rules.md`.
- Open owner or external-system questions live in `docs/agent-native/owner-questions.md`.
- Former `docs/context/` topic docs were merged or moved under
  `docs/agent-native/` on 2026-05-11 so future-agent loading has one durable
  root.

## Legacy AI Or Skill-System Docs

No substantial legacy AI-doc or old skill-system doc tree was found in this
checkout on 2026-05-11. The new `docs/agent-native/` tree is the canonical
future-agent doc surface.

## Missing Docs

| Needed doc or answer | Why needed | Owner/question |
| --- | --- | --- |
| External benchmark dependency policy | SWE-bench needs optional package and Docker; local smoke avoids both | See `owner-questions.md`. |
| Release ownership after open-source prep | `bf026ac` prepared the first open-source release, but reviewer/release owner flow is not documented beyond CI | See `owner-questions.md`. |

No separate `repo-map.md` is needed while normal work remains self-contained in
`/Users/bytedance/Documents/simple_agent`. Add one only if a future task makes a
related repo or external system load-bearing.

The first live provider adapter target is no longer a missing answer: owner
confirmation on 2026-05-11 chose `openai-chat`.
