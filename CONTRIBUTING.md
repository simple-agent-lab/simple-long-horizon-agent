# Contributing

Thanks for your interest in Simple Agent Lab. This is a small, docs-first
project — the goal is to keep the agent loop readable and modifiable by
students, small teams, and learners. Contributions that preserve that goal
are very welcome.

## Before you start

Read these in order:

1. [`README.md`](README.md) — what this project is and how to set it up.
2. [`AGENTS.md`](AGENTS.md) — collaboration contract for humans and coding
   agents (working principles, goals, non-goals, editing expectations).
3. [`docs/agent-native/development.md`](docs/agent-native/development.md) — day-to-day
   commands and the quality gate.

If you are introducing an architectural commitment, add a decision record
under [`docs/decisions/`](docs/decisions/) following the existing template.

## Local quality gate

Before opening a pull request, the same checks that GitHub Actions runs
should pass locally:

```bash
bash runs/run_ci.sh
```

This runs `ruff format --check .` (format check), `ty check src` (type check),
and `python -m unittest discover -s tests/unit` (the unit suite) on your local
Python. All must exit `0`. CI runs the same gate against Python 3.10 through
3.13.

If you add a new dev dependency, put it under `[dependency-groups] dev` in
`pyproject.toml`, not `[project] dependencies`. The runtime is supposed to
install with zero third-party deps.

## Pull requests

- Keep changes small and focused. One commit per logical change is preferred.
- Update the relevant README, ADR, or `docs/agent-native/` note in the same PR
  when behavior or contracts change.
- If your change touches an area covered by an ADR, link the ADR in the PR
  description and call out whether it confirms, extends, or supersedes the
  decision.
- Use the PR template in [`.github/pull_request_template.md`](.github/pull_request_template.md).

## Reporting bugs and proposing features

Use the GitHub issue templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). For security-sensitive
reports, prefer a private channel — see the project owner's email in
`pyproject.toml` rather than filing a public issue.

## Licensing of contributions

This project is licensed under the
[Apache License, Version 2.0](LICENSE). Section 5 of that license states
that any contribution you intentionally submit for inclusion in the work is
licensed under the same terms, unless you explicitly state otherwise. By
opening a pull request you confirm that you have the right to contribute the
code under those terms.

No separate CLA or DCO sign-off is required at this time.
