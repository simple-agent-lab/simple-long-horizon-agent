export const meta = {
  name: 'simplicity-review',
  description: 'Lean per-MR code review for simplicity & code quality — 2 parallel reviewers, 1 batch verify pass',
  whenToUse: 'Review an MR/branch diff for simplicity, reuse, readability, and code-quality cleanups (not a bug hunt). Pass a git range as args (e.g. "origin/main...HEAD"); defaults to the diff against origin/main.',
  phases: [
    { title: 'Review', detail: 'two reviewers in parallel: complexity+reuse / readability+style' },
    { title: 'Verify', detail: 'one pass: drop churn, behavior changes, convention fights' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'file', 'line', 'severity', 'problem', 'suggestion'],
        properties: {
          title: { type: 'string', description: 'short imperative title' },
          file: { type: 'string' },
          line: { type: 'string', description: 'line number or range' },
          severity: { type: 'string', enum: ['minor', 'moderate', 'significant'] },
          problem: { type: 'string' },
          suggestion: { type: 'string', description: 'concrete simpler alternative' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      description: 'one entry per finding, in the same order',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['index', 'keep', 'reason'],
        properties: {
          index: { type: 'integer', description: '0-based index of the finding being judged' },
          keep: { type: 'boolean', description: 'true only if real, behavior-preserving, and genuinely simpler' },
          reason: { type: 'string' },
        },
      },
    },
  },
}

const range = (typeof args === 'string' && args.trim()) ? args.trim() : 'origin/main...HEAD'

const REVIEWERS = [
  {
    key: 'simplicity',
    prompt: `Lens: SIMPLICITY & REUSE. Flag unnecessary complexity and over-abstraction (single-use helpers,
premature generalization, needless layering, collapsible control flow) and duplication (logic that
re-implements an existing helper/stdlib/sibling pattern, copy-paste, unifiable branches).`,
  },
  {
    key: 'quality',
    prompt: `Lens: READABILITY, DEAD CODE & IDIOM. Flag unclear names, redundant/misleading comments, deep
nesting, unused vars/params/imports, redundant checks, wasted/repeated work, and deviations from the
conventions used elsewhere in THIS codebase (error handling, naming, construction, test style).`,
  },
]

phase('Review')
const reviews = await parallel(REVIEWERS.map(r => () =>
  agent(
    `Review the diff \`git diff ${range}\` through ONE lens only.

${r.prompt}

Use bash/git and read surrounding files to ground every finding — verify any "existing helper" or
"convention" you cite really exists. Don't re-run the full test suite or a full build — CI handles that —
but when you're unsure about a piece of code or a dependency, you MAY run quick, targeted checks to verify
(e.g. confirm a symbol/helper exists, a focused type-check or vet/build on the affected package). Only report
changes WITHIN this diff, each pointing at a real line with a concrete simpler alternative. This is a QUALITY
review: do NOT report correctness bugs. If the diff is already clean through this lens, return an empty
findings array.`,
    { label: `review:${r.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, agentType: 'Explore' },
  ).then(res => res?.findings ?? []),
))

// merge + dedup by file:line — both reviewers can land on the same spot
const seen = new Set()
const findings = reviews.flat().filter(f => {
  const k = `${f.file}:${f.line}`
  if (seen.has(k)) return false
  seen.add(k)
  return true
})

if (findings.length === 0) {
  log(`No quality findings in ${range}.`)
  return { range, findings: [] }
}

phase('Verify')
const list = findings.map((f, i) =>
  `[${i}] ${f.title} — ${f.file}:${f.line}\n    problem: ${f.problem}\n    suggestion: ${f.suggestion}`).join('\n')

const { verdicts } = await agent(
  `Adversarially verify these code-quality findings from \`git diff ${range}\`. Read the actual files. Don't
re-run the full test suite or build (CI handles that), but when unsure about code or a dependency you MAY run
quick, targeted checks to confirm. Be skeptical: drop churn-for-churn, anything that changes behavior, that
fights a deliberate local convention, or that merely relocates complexity. Default to keep=false when
uncertain. Return one verdict per finding.

${list}`,
  { label: 'verify-batch', phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'Explore' },
)

const keep = new Set(verdicts.filter(v => v.keep).map(v => v.index))
const rank = { significant: 0, moderate: 1, minor: 2 }
const confirmed = findings
  .filter((_, i) => keep.has(i))
  .sort((a, b) => (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3))

log(`${findings.length} finding(s) → ${confirmed.length} confirmed.`)

return {
  range,
  rawFindings: findings.length,
  confirmed: confirmed.map(f => ({
    title: f.title,
    location: `${f.file}:${f.line}`,
    severity: f.severity,
    problem: f.problem,
    suggestion: f.suggestion,
  })),
}
