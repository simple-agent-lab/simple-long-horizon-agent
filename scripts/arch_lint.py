"""Lint architectural boundaries so the runtime stays simple and provider-neutral.

The rules are declarative and live at the top of this file. They encode three
invariants that the codebase already satisfies; this script keeps them true:

1. Layering. Every internal module belongs to a zone with a rank. A module may
   only import modules in its own zone or a lower one. In particular the
   foundation (`messages`) imports nothing internal, and the runtime core never
   depends on peripheral subsystems (agents, evals, mcp, trace, gateway).
2. Provider isolation. Only provider adapters may import a provider SDK, so the
   runtime core stays provider-neutral.
3. Optional-dependency confinement. Heavy optional dependencies stay inside the
   one subpackage that owns them, so `import simple_agent_lab` needs none of
   them.

Usage:
    python scripts/arch_lint.py
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "simple_agent_lab"
PACKAGE_ROOT = ROOT / "src" / PACKAGE

# --- Rules ----------------------------------------------------------------
#
# Zones are ranked. An internal import is allowed only when the target zone's
# rank is <= the importer zone's rank (same zone or lower). `api` is the public
# assembly point (the package __init__) and may import anything.

FOUNDATION, CORE, PERIPHERAL, API = "foundation", "core", "peripheral", "api"

ZONE_RANK = {FOUNDATION: 0, CORE: 1, PERIPHERAL: 2, API: 3}

# Top-level module / subpackage name -> zone. Every top-level name under the
# package must appear here; a new one is a lint error until it is placed
# deliberately into a zone.
MODULE_ZONES = {
    # foundation: the message protocol, the root of everything
    "messages": FOUNDATION,
    # core runtime: orchestration, state, context, model boundary, tools
    "protocols": CORE,
    "context_view": CORE,
    "compression": CORE,
    "state": CORE,
    "core": CORE,
    "hooks": CORE,
    "llm_agent": CORE,
    "llm": CORE,
    "tools": CORE,
    # peripheral subsystems: may depend on core, but core must not depend back
    "agents": PERIPHERAL,
    "trace": PERIPHERAL,
    "evals": PERIPHERAL,
    "mcp": PERIPHERAL,
    "tui_gateway": PERIPHERAL,
    "skills": PERIPHERAL,
}

# External (third-party) top-level package -> the only internal path prefix
# allowed to import it (dotted, relative to the package root).
RESTRICTED_EXTERNAL = {
    "anthropic": "llm.adapters",
    "openai": "llm.adapters",
    "docker": "evals.backends",
    "mcp": "mcp",
    "datasets": "evals",
    "swebench": "evals",
}

# Top-level packages the shipped library must never import. Tests, examples, and
# scripts are scaffolding that may depend on the library, never the reverse.
FORBIDDEN_EXTERNAL = ("tests", "examples", "scripts", "studio")


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def module_dotted_name(path: Path) -> str:
    """Dotted module name relative to the package root (``__init__`` collapses)."""
    rel = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def module_package(dotted: str, is_init: bool) -> str:
    """The package a module lives in, used to resolve relative imports."""
    if is_init:
        return dotted
    return dotted.rsplit(".", 1)[0] if "." in dotted else ""


def zone_of(dotted: str) -> str | None:
    """Zone of an internal module, or None if its top-level name is unclassified."""
    if dotted == "":
        return API
    top = dotted.split(".", 1)[0]
    return MODULE_ZONES.get(top)


def resolve_relative(package: str, level: int, module: str | None) -> str:
    """Resolve a relative import to a dotted name within the package root."""
    base_parts = package.split(".") if package else []
    if level > 1:
        base_parts = base_parts[: -(level - 1)] if level - 1 <= len(base_parts) else []
    if module:
        base_parts = base_parts + module.split(".")
    return ".".join(base_parts)


def is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def iter_runtime_imports(tree: ast.AST):
    """Yield import statements, skipping bodies guarded by ``if TYPE_CHECKING``."""

    def walk(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node
            elif isinstance(node, ast.If) and is_type_checking_guard(node):
                yield from walk(node.orelse)  # the else branch still runs
            else:
                for child in ast.iter_child_nodes(node):
                    yield from walk([child])

    yield from walk(list(getattr(tree, "body", [])))


def imported_targets(node: ast.AST, package: str):
    """Yield (kind, name, line) for one import node.

    kind is "internal" with a dotted package-relative name, or "external" with a
    top-level third-party package name.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".", 1)[0]
            if top == PACKAGE:
                yield "internal", alias.name[len(PACKAGE) + 1 :], node.lineno
            else:
                yield "external", top, node.lineno
        return
    # ImportFrom
    if node.level > 0:
        yield (
            "internal",
            resolve_relative(package, node.level, node.module),
            node.lineno,
        )
        return
    module = node.module or ""
    top = module.split(".", 1)[0]
    if top == PACKAGE:
        yield "internal", module[len(PACKAGE) + 1 :], node.lineno
    elif top:
        yield "external", top, node.lineno


def under_prefix(dotted: str, prefix: str) -> bool:
    return dotted == prefix or dotted.startswith(prefix + ".")


def lint_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    dotted = module_dotted_name(path)
    package = module_package(dotted, path.name == "__init__.py")

    src_zone = zone_of(dotted)
    violations: list[Violation] = []
    if src_zone is None:
        top = dotted.split(".", 1)[0]
        return [
            Violation(
                path,
                1,
                f"module '{top}' is not assigned to a zone; add it to MODULE_ZONES "
                f"in scripts/arch_lint.py",
            )
        ]

    for node in iter_runtime_imports(tree):
        for kind, name, line in imported_targets(node, package):
            if kind == "external":
                if name in FORBIDDEN_EXTERNAL:
                    violations.append(
                        Violation(
                            path,
                            line,
                            f"the library must not import '{name}'; tests, "
                            f"examples, and scripts depend on the library, not "
                            f"the reverse",
                        )
                    )
                    continue
                prefix = RESTRICTED_EXTERNAL.get(name)
                if prefix is not None and not under_prefix(dotted, prefix):
                    violations.append(
                        Violation(
                            path,
                            line,
                            f"'{name}' may only be imported under {PACKAGE}.{prefix} "
                            f"(keeps the runtime core free of this dependency)",
                        )
                    )
                continue

            target_zone = zone_of(name)
            if target_zone is None:
                continue  # unclassified target is reported where that file is linted
            if ZONE_RANK[target_zone] > ZONE_RANK[src_zone]:
                violations.append(
                    Violation(
                        path,
                        line,
                        f"{src_zone} module '{dotted or PACKAGE}' must not import "
                        f"{target_zone} module '{name}' "
                        f"(a {src_zone} module may only depend on {src_zone} or lower)",
                    )
                )

    return violations


def package_files() -> list[Path]:
    # The bundled skill library (skills/library/) ships third-party reference
    # scripts the agent reads and runs in a sandbox; they are shipped assets,
    # not part of this package's import graph, so they are not architecture-linted.
    library = PACKAGE_ROOT / "skills" / "library"
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and library not in p.parents
    )


def main() -> int:
    violations: list[Violation] = []
    for path in package_files():
        violations.extend(lint_file(path))

    if violations:
        print("Architecture lint failed:")
        for v in violations:
            print(f"  {v.format()}")
        return 1

    print("Architecture lint passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
