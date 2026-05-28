---
target: studio/trace-viewer/index.html
total_score: 20
p0_count: 0
p1_count: 3
timestamp: 2026-05-27T12-26-30Z
slug: tools-trace-viewer-index-html
---
# Critique — tools-trace-viewer-index-html

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | No confirmation that a dropped file loaded. |
| 2 | Match System / Real World | 3 | "OBS · 001" + producer/schema/trace_id assume domain fluency. |
| 3 | User Control and Freedom | 1 | No clear-filters, no row deselect, no reload-last-file, no hide-inspector. |
| 4 | Consistency and Standards | 3 | Stat strip mixes Fraunces and JetBrains Mono row-to-row; depth indentation math differs between tree and stream. |
| 5 | Error Prevention | 2 | Two alert() paths for bad files. No schema validation. Latent NaN when before_tokens === 0. |
| 6 | Recognition Rather Than Recall | 3 | Legend strip + labels good. Glyph meanings have no tooltip. |
| 7 | Flexibility and Efficiency | 1 | Zero keyboard shortcuts. No jump-to-next-error. |
| 8 | Aesthetic and Minimalist Design | 2 | 10-cell stat strip > working-memory limit. Decorative pulse + grain dont earn their pixel. |
| 9 | Error Recovery | 1 | alert() only path; no error boundary. |
| 10 | Help and Documentation | 1 | README outside the page; no inline tooltips, no ?, no first-run hint for drag-and-drop. |
| **Total** | | **20/40** | **Acceptable — solid bones, weak power-user surface** |

## Anti-Patterns Verdict

LLM: passes first-order AI-slop check (committed type system, distinct palette, frame-counter index). Fails second-order: sits in a saturated "editorial-typographic dark-mode developer tool" family.

Deterministic scan: 4 findings (3 false-positive low-contrast against assumed white background; 1 true-positive: Fraunces on the overused-font list).

Author-detected, missed by scan:
- Side-stripe borders > 1px (absolute ban) on .content-block (3px), .row.depth-1/.depth-2 (2px), .node.selected (2px).
- Display serif on UI data/labels (product-register ban): Fraunces italic on stat integers, inspector h3, thinking-block text.
- Em dashes in copy (parent ban): README headings, <title>, runtime placeholders.
- Hero-metric template (borderline).

Visual overlay: unavailable (local detector engine missing; sandbox can't reach localhost). Fallback: source review + headless screenshots.

## Overall Impression

Good first impression that punishes the second sitting. The IA is honest (each view maps a trace layer). The shipping-blocker is the missing power-user keyboard layer; the decorative type choices are a smaller, separate problem. The single biggest opportunity is wiring the anomaly indicators together so the eye finds failure before the user has to scan for it.

## What's Working

1. Three view modes (messages / events / model turns) map 1:1 to the trajectory module's Event → Span → ModelTurn layers; re-derives in the browser so older records work.
2. Sub-agent visualization is consistent across tree, stream, and waterfall, all magenta-colored.
3. Stat strip surfaces failure modes specific to this runtime (compressions, sub-agents, exit reason), not a generic dashboard template.

## Priority Issues

- [P1] Zero keyboard navigation. Fix: global keymap (j/k row nav, / search, 1/2/3 modes, e errors-only, n next-error, esc deselect, ? overlay). Command: harden.
- [P1] Side-stripe borders > 1px in three places (sub-agent depth, content-block kind, row selection). Fix: replace with leading semantic glyphs + faint background tints + outline-only selection. Command: polish.
- [P1] Display serif on data and labels (product-register ban). Fix: limit Fraunces to one place (page-title eyebrow); move stat values, inspector h3, section labels, thinking-block text to Plex Sans or JetBrains Mono. Command: typeset.
- [P2] Anomaly indicators isolated; same fact rendered four times, no cross-navigation. Fix: clickable red stat scrolls to first error and selects across panes; add next/prev anomaly keys. Command: craft.
- [P2] No focus indicators on most interactive elements; meaning carried by color alone; grain overlay above content; no reduced-motion fallback. Fix: :focus-visible rule, drop grain z-index, wrap pulse in prefers-reduced-motion. Command: audit.

## Persona Red Flags

Alex (Power User): no keyboard shortcuts; no jump-to-anomaly; substring-only search; no recent-traces; no trace diff.
Sam (Accessibility): no visible focus on chips/toggle/tree/rows; grain overlay above content; error semantics color-only; 9.5–10px label text; pulsing dot has no reduced-motion fallback; no aria-live for selection.
Riley (Stress Tester): trace missing events crashes setTrace; before_tokens=0 renders NaN% label; waterfall has no horizontal scroll on dense traces; README mentions regex but it's substring only; tool_call_id collisions silently break sub-agent mapping.
Self-hosting agent developer: tool shows what happened but doesn't connect the failure dots; user will guess instead of learn.

## Minor Observations

- ▸ glyph on tree nodes suggests collapse but is decorative.
- Empty inspector state could teach the click targets and (future) keyboard equivalents.
- Mode toggle and filter chips use different shapes despite both being chip-style; pick one vocabulary.
- Stream count noun jumps between roles ("messages · 14", "events · N").
- Drop-zone hint only visible during drag; add a persistent quiet hint.
- <title> em dash + all-caps brand voice in a tool tab.

## Questions to Consider

- What if the eye landed on the failed span first, the layout scrolling the stream and waterfall to it on load?
- Does the tool need three view modes, or could events / model turns be progressive disclosures inside messages?
- If Fraunces were removed entirely, would the design lose its voice or gain its honesty?
- Would a single-column reading mode (no inspector) better serve "read this trace like a story"?
- Is the inspector the right home for raw JSON, or should that live in a dedicated drawer?
