from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.skills.discovery import (
    BUNDLED_LIBRARY_DIR,
    SkillMetadata,
    SkillRoot,
    default_skill_roots,
    discover_skills,
    load_skill_from_file,
    parse_frontmatter,
)


def _write_skill(root: Path, name: str, description: str = "does a thing") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    body = (
        f"---\nname: {name}\ndescription: {description}\n---\n"
        f"# {name}\nDo the {name} workflow.\n"
    )
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir / "SKILL.md"


class DiscoveryTest(unittest.TestCase):
    def test_parse_frontmatter_reads_name_and_description(self) -> None:
        fm, body = parse_frontmatter(
            "---\nname: pdf-tools\ndescription: rotate PDFs\n---\nbody here\n"
        )
        self.assertEqual(fm["name"], "pdf-tools")
        self.assertEqual(fm["description"], "rotate PDFs")
        self.assertEqual(body, "body here\n")

    def test_parse_frontmatter_without_frontmatter_returns_empty(self) -> None:
        fm, body = parse_frontmatter("no frontmatter here")
        self.assertEqual(fm, {})
        self.assertEqual(body, "no frontmatter here")

    def test_load_skill_skips_missing_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x" / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text("---\nname: x\n---\nbody\n", encoding="utf-8")
            self.assertIsNone(load_skill_from_file(str(path), "repo"))

    def test_load_skill_returns_metadata_with_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_skill(Path(tmp), "alpha")
            skill = load_skill_from_file(str(md), "repo")
        assert skill is not None
        self.assertEqual(skill.name, "alpha")
        self.assertEqual(skill.scope, "repo")
        self.assertTrue(skill.path_to_skill_md.endswith("alpha/SKILL.md"))
        self.assertTrue(skill.base_dir.endswith("alpha"))

    def test_read_body_loads_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_skill(Path(tmp), "beta")
            skill = load_skill_from_file(str(md), "repo")
            assert skill is not None
            self.assertIn("beta workflow", skill.read_body())

    def test_discover_finds_skills_under_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            _write_skill(root, "one")
            _write_skill(root, "two")
            skills = discover_skills([SkillRoot(str(root), "repo")])
        self.assertEqual(sorted(s.name for s in skills), ["one", "two"])

    def test_discover_dedups_and_resolves_collisions_by_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            user_root = Path(tmp) / "user"
            _write_skill(repo_root, "dup", description="repo version")
            _write_skill(user_root, "dup", description="user version")
            skills = discover_skills(
                [SkillRoot(str(user_root), "user"), SkillRoot(str(repo_root), "repo")]
            )
        self.assertEqual(len(skills), 1)
        # repo outranks user, so the repo description wins.
        self.assertEqual(skills[0].description, "repo version")

    def test_default_roots_include_bundled_library_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = default_skill_roots(tmp, home=tmp)
        self.assertEqual(roots[0].path, BUNDLED_LIBRARY_DIR)
        self.assertEqual(roots[0].scope, "bundled")
        scopes = {r.scope for r in roots}
        self.assertEqual(scopes, {"bundled", "repo", "user"})

    def test_default_roots_walk_up_to_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / ".git").mkdir()
            nested = base / "a" / "b"
            nested.mkdir(parents=True)
            roots = default_skill_roots(str(nested), home=tmp)
        repo_paths = [r.path for r in roots if r.scope == "repo"]
        # Walks up from nested through the git root, then stops.
        self.assertTrue(any(p.endswith("a/b/.agents/skills") for p in repo_paths))
        self.assertTrue(
            any(p.endswith(str(base / ".agents/skills")) for p in repo_paths)
        )


from simple_agent_lab.skills.prompt import (  # noqa: E402
    SKILLS_HOW_TO_USE,
    render_skills_instructions,
    skills_menu_message,
)


class PromptMenuTest(unittest.TestCase):
    def _skill(self, name: str) -> SkillMetadata:
        return SkillMetadata(
            name=name,
            description=f"{name} description",
            path_to_skill_md=f"/abs/{name}/SKILL.md",
            base_dir=f"/abs/{name}",
            scope="repo",
        )

    def test_render_empty_when_no_skills(self) -> None:
        self.assertEqual(render_skills_instructions([]), "")

    def test_render_lists_each_skill_with_path(self) -> None:
        block = render_skills_instructions([self._skill("alpha"), self._skill("beta")])
        self.assertIn("<skills_instructions>", block)
        self.assertIn("</skills_instructions>", block)
        self.assertIn("- alpha: alpha description (file: /abs/alpha/SKILL.md)", block)
        self.assertIn("- beta: beta description (file: /abs/beta/SKILL.md)", block)
        self.assertIn("### How to use skills", block)
        self.assertIn(SKILLS_HOW_TO_USE.strip().splitlines()[0], block)

    def test_menu_message_is_a_system_message(self) -> None:
        msg = skills_menu_message([self._skill("alpha")], target="agent")
        assert msg is not None
        self.assertEqual(msg.role, "system")
        self.assertEqual(msg.kind, "system")
        self.assertEqual(msg.target, "agent")
        self.assertIn("alpha description", msg.content[0].text)

    def test_menu_message_none_when_no_skills(self) -> None:
        self.assertIsNone(skills_menu_message([], target="agent"))


from simple_agent_lab.skills.directives import (  # noqa: E402
    SkillDirectives,
    parse_skill_directives,
)


class DirectivesTest(unittest.TestCase):
    def test_plain_task_keeps_skills_enabled(self) -> None:
        d = parse_skill_directives("fix the bug in parser.py", {"alpha"})
        self.assertTrue(d.skills_enabled)
        self.assertEqual(d.mentions, ())
        self.assertEqual(d.cleaned_task, "fix the bug in parser.py")

    def test_no_skills_directive_disables_and_is_stripped(self) -> None:
        d = parse_skill_directives("/no-skills just answer directly", set())
        self.assertFalse(d.skills_enabled)
        self.assertEqual(d.cleaned_task, "just answer directly")

    def test_no_skills_directive_mid_text(self) -> None:
        d = parse_skill_directives("do the thing /no-skills please", set())
        self.assertFalse(d.skills_enabled)
        self.assertEqual(d.cleaned_task, "do the thing please")

    def test_mention_resolves_known_skill(self) -> None:
        d = parse_skill_directives("use $alpha to do it", {"alpha", "beta"})
        self.assertEqual(d.mentions, ("alpha",))
        # The $mention stays in the task text (the model sees it, per Codex).
        self.assertIn("$alpha", d.cleaned_task)

    def test_unknown_mention_ignored(self) -> None:
        d = parse_skill_directives("use $nope", {"alpha"})
        self.assertEqual(d.mentions, ())

    def test_env_var_lookalike_ignored(self) -> None:
        d = parse_skill_directives("echo $PATH and $HOME", {"alpha"})
        self.assertEqual(d.mentions, ())

    def test_duplicate_mentions_collapse(self) -> None:
        d = parse_skill_directives("$alpha then $alpha again", {"alpha"})
        self.assertEqual(d.mentions, ("alpha",))

    def test_returns_skill_directives_type(self) -> None:
        self.assertIsInstance(parse_skill_directives("x", set()), SkillDirectives)


from simple_agent_lab.agents.bash import make_bash_agent  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402
from simple_agent_lab.skills.runtime import (  # noqa: E402
    run_with_skills,
    skill_body_messages,
)

FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


class RunWithSkillsTest(unittest.TestCase):
    def _fixture_roots(self) -> list[SkillRoot]:
        return [SkillRoot(str(FIXTURES), "repo")]

    def test_skill_body_messages_wrap_body(self) -> None:
        skills = discover_skills(self._fixture_roots())
        msgs = skill_body_messages(skills, target="agent")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[0].kind, "context")
        text = msgs[0].content[0].text
        self.assertIn("<skill>", text)
        self.assertIn("<name>echo-fixture</name>", text)
        self.assertIn("To echo text", text)

    def test_menu_is_recorded_before_task(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=str(FIXTURES))
        state, events = run_with_skills(
            agent,
            "echo the word hello",
            roots=self._fixture_roots(),
            max_turns=3,
        )
        for _ in events:
            pass
        kinds = [m.kind for m in state.messages]
        self.assertIn("system", kinds)
        menu = next(m for m in state.messages if m.kind == "system")
        self.assertIn("echo-fixture", menu.content[0].text)
        # Menu precedes the task message.
        self.assertLess(
            state.messages.index(menu),
            next(i for i, m in enumerate(state.messages) if m.kind == "task"),
        )

    def test_mention_injects_body_before_task(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=str(FIXTURES))
        state, events = run_with_skills(
            agent,
            "use $echo-fixture to echo hello",
            roots=self._fixture_roots(),
            max_turns=3,
        )
        for _ in events:
            pass
        context_msgs = [m for m in state.messages if m.kind == "context"]
        self.assertTrue(context_msgs)
        self.assertIn("<name>echo-fixture</name>", context_msgs[0].content[0].text)

    def test_no_skills_directive_suppresses_everything(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=str(FIXTURES))
        state, events = run_with_skills(
            agent,
            "/no-skills just echo hello",
            roots=self._fixture_roots(),
            max_turns=3,
        )
        for _ in events:
            pass
        self.assertNotIn(
            "system", [m.kind for m in state.messages if m.sender == "skills"]
        )
        self.assertFalse([m for m in state.messages if m.kind == "context"])
        task = next(m for m in state.messages if m.kind == "task")
        self.assertEqual(task.content[0].text, "just echo hello")

    def test_no_skills_when_no_skills_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=empty)
            state, events = run_with_skills(
                agent, "do something", roots=[SkillRoot(empty, "repo")], max_turns=3
            )
            for _ in events:
                pass
            self.assertNotIn("system", [m.kind for m in state.messages])
            self.assertNotIn("context", [m.kind for m in state.messages])

    def test_preload_injects_body_without_mention(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=str(FIXTURES))
        state, events = run_with_skills(
            agent,
            "echo hello",  # no $mention
            roots=self._fixture_roots(),
            preload=["echo-fixture"],
            max_turns=3,
        )
        for _ in events:
            pass
        context_msgs = [m for m in state.messages if m.kind == "context"]
        self.assertTrue(context_msgs)
        self.assertIn("<name>echo-fixture</name>", context_msgs[0].content[0].text)

    def test_injected_body_includes_file_manifest(self) -> None:
        skills = discover_skills(self._fixture_roots())
        text = skill_body_messages(skills, target="agent")[0].content[0].text
        self.assertIn("<files>", text)
        # The fixture bundles scripts/echo.py; SKILL.md itself is excluded.
        self.assertIn("scripts/echo.py", text)
        self.assertNotIn("<files>\nSKILL.md", text)


if __name__ == "__main__":
    unittest.main()
