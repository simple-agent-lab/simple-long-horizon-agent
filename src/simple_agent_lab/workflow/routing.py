"""Routing (dispatcher).

A lightweight *router* classifies the task and hands it to the single best
specialist agent. Use it when you have several focused agents (e.g. "sql",
"python", "prose") and want one cheap up-front decision instead of giving
one generalist every tool.

The router and each specialist are ordinary agents driven by the core ReAct
loop. The workflow runs the router, parses its choice against the known
routes, then runs the chosen specialist on the original task.

Note the contrast with `core.task_tool`: there the *parent model* picks a
sub-agent mid-loop via a tool call; here a dedicated router agent makes the
choice as a separate, inspectable step before any specialist runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AbortFlag

from .base import WorkflowResult, as_text, never_abort, run_agent

ROUTER_ROLE = "Pick the single best specialist for a task."


@dataclass(frozen=True)
class Route:
    """One routing destination: a `name` the router can pick and its agent."""

    name: str
    agent: Agent
    description: str = ""


def _router_system_prompt(routes: Sequence[Route]) -> str:
    listing = "\n".join(
        f"- {route.name}: {route.description or route.agent.role or '(no description)'}"
        for route in routes
    )
    names = ", ".join(route.name for route in routes)
    return (
        "You are a routing agent. Read the task and choose the single best "
        "specialist to handle it from the list below. Reply with ONLY that "
        "specialist's name on its own line — no explanation.\n\n"
        f"Specialists:\n{listing}\n\n"
        f"Valid names: {names}"
    )


def _router_prompt(task: str, routes: Sequence[Route]) -> str:
    names = ", ".join(route.name for route in routes)
    return (
        f"Choose the best specialist for this task (one of: {names}).\n\nTask:\n{task}"
    )


def select_route(
    router_output: str,
    routes: Sequence[Route],
    *,
    default: str | None = None,
) -> Route | None:
    """Resolve the router's free text to one `Route`.

    Tries an exact name match (case-insensitive, ignoring surrounding
    whitespace/punctuation) first, then falls back to the longest route name
    that appears as a substring — longest-first so "sql_writer" wins over a
    bare "sql" when both are mentioned. Returns the `default` route if nothing
    matches, or `None` if there is no usable choice.
    """
    cleaned = router_output.strip().strip(".\"'` ").lower()
    by_name = {route.name.lower(): route for route in routes}

    if cleaned in by_name:
        return by_name[cleaned]

    haystack = router_output.lower()
    for route in sorted(routes, key=lambda r: len(r.name), reverse=True):
        if route.name.lower() in haystack:
            return route

    if default is not None and default.lower() in by_name:
        return by_name[default.lower()]
    return None


def run_routing(
    router: Agent,
    routes: Sequence[Route],
    task: str,
    *,
    router_max_turns: int = 4,
    worker_max_turns: int = 20,
    default: str | None = None,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Route `task` to one specialist and run it.

    Runs `router` to pick a route, resolves the pick with `select_route`
    (falling back to `default`), then runs the chosen specialist on the
    original task. The result's `steps` are `[router, specialist]`. If no
    route can be resolved, the result holds just the router step and its
    output is the router's text (so the caller can see what went wrong).
    """
    route_list = list(routes)
    if not route_list:
        raise ValueError("run_routing requires at least one route")

    task_text = as_text(task)
    router_step = run_agent(
        router,
        _router_prompt(task_text, route_list),
        max_turns=router_max_turns,
        abort=abort,
        role="router",
    )

    chosen = select_route(router_step.output, route_list, default=default)
    if chosen is None or abort():
        return WorkflowResult(output=router_step.output, steps=[router_step])

    worker_step = run_agent(
        chosen.agent,
        task_text,
        max_turns=worker_max_turns,
        abort=abort,
        role=f"route:{chosen.name}",
    )
    return WorkflowResult(output=worker_step.output, steps=[router_step, worker_step])


def make_router_agent(
    provider: LLMProvider,
    routes: Sequence[Route],
    *,
    name: str = "router",
    role: str = ROUTER_ROLE,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build a router `Agent` whose system prompt lists `routes`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=_router_system_prompt(routes),
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
