# Bundled skill library

This directory ships inside the `simple-agent-lab` wheel and is scanned for
skills by default (see `simple_agent_lab.skills.discovery.default_skill_roots`).

It is intentionally empty of skills: the skills *system* is on by default, but
no skill is advertised until one is added here or to a project/user skill root
(`.agents/skills/<name>/SKILL.md`). Drop a directory containing a `SKILL.md`
here to ship a default skill with the package.

A skill directory looks like:

```text
my-skill/
  SKILL.md        # required: YAML frontmatter (name, description) + body
  scripts/        # optional: runnable helpers (model runs them via bash)
  references/     # optional: extra docs (model reads them via the read tool)
```
