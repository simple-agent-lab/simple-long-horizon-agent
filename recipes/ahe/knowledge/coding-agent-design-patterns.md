# Coding Agent Design Patterns

This note captures reusable patterns for coding agents in the AHE spirit.
Keep it general. Do not turn it into a leaderboard note or a live-web summary.

## Goals

- Make the agent's plan visible.
- Keep edits small enough to inspect.
- Preserve evidence for later review.
- Separate what changed from why it changed.

## Helpful patterns

1. Start with a narrow surface.
   Give the agent a small set of files or components it can change.

2. Attach evidence to every proposal.
   The agent should cite runs, failures, or observations instead of relying on
   vague intent.

3. Analyze before editing.
   A short pre-proposal analysis helps the next change stay grounded in the
   observed failures.

4. Keep the change manifest explicit.
   Name the touched component, the expected effect, and the risks.

5. Log outcomes in append-only form.
   Human readers should be able to reconstruct the run without replaying it.

6. Prefer measurable deltas over rhetorical confidence.
   Use task history, reward means, or other concrete signals when possible.

7. Keep the artifact tree navigable.
   Round directories, manifest files, and summaries should be easy to scan.

8. Make fallback behavior obvious.
   If a model output is missing or unusable, write down what happened.

9. Use the smallest component that fits the change.
   The best edit is often the one that touches less, not more.

10. Separate recipe policy from framework machinery.
    The substrate should stay generic; the recipe can own domain choices.

## Review heuristics

- Ask whether the change can be explained in one paragraph.
- Ask whether the run output would let another person verify it later.
- Ask whether the change surface is broader than the evidence requires.
- Ask whether a missing artifact would make the run hard to trust.

## Common failure modes

- Over-editing because the surface is too broad.
- Hiding proposal logic inside ad hoc scripts.
- Mixing analysis, decision, and score data into one blob.
- Claiming improvement from dry-run wiring alone.
- Letting strategy prompts grow so large they stop being inspectable.

## AHE-flavored habits that generalize well

- Observe the current system before proposing a rewrite.
- Record the decision context next to the decision.
- Keep the evaluation path separate from the explanation path.
- Use names that match the artifact's job.
- Prefer one clear artifact per concern.

## When to be cautious

- If a pattern depends on live external data, document the dependency clearly.
- If a pattern hides the reason for an edit, simplify it.
- If a pattern makes review harder than execution, it is probably too clever.
- If the pattern only works for one benchmark, keep it recipe-local.

## Practical rule of thumb

When in doubt, choose the path that would make a future reader say:
"I can see what happened, and I can see why."

