# Human docs

Plain-language, self-contained explainers written for people — read each on its
own, no jumping between files. When a concept is easier to grasp as a visual
walkthrough than as a terse decision record or an API reference, it belongs here.

## Contents

- `integrating-a-bench.html` — a visual, self-contained guide to the standard
  way to integrate a new benchmark on this framework: the two halves you write
  (host `Suite` + container functions), the five steps, what the framework does
  in between, and common pitfalls. Open it in a browser.
- `running-offline.html` — how to run container evals with no PyPI access:
  build a wheelhouse from `uv.lock`, hand it (and, for pre-3.11 images, a Linux
  `uv` binary) to `LocalDockerBackend` as read-only bind mounts, and let the
  bootstrap install with `--no-index`. Covers the shared-filesystem assumption
  and the common bind-mount/ABI pitfalls. Open it in a browser.
- `self-evolution-design.zh.html` — the design-review snapshot for the
  evolution framework (Chinese): how the design converged, the literature and
  competitor landscape it absorbed, the three-nouns/two-verbs architecture,
  the task/meta update mechanics, the implemented skeleton, the roadmap, and
  the concrete questions reviewers should weigh in on. The living design lives
  in `docs/design/`; this page is the team-review companion. Open it in a
  browser.

The two guides have a Chinese version alongside (`*.zh.html`) with a language
switcher in the header; the design-review page is currently Chinese-only.
