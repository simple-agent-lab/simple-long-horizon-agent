"""Container-local runner for the Simple Agent Lab Harbor agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from simple_agent_lab.agent_flavors import SIMPLE_AGENT_FLAVORS
from simple_agent_lab.agents.starter import agent_session
from simple_agent_lab.evals.harbor import DEFAULT_API_KIND, DEFAULT_MAX_TURNS
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    FAKE_PROVIDER,
    OPENAI_ENV,
    provider_from_env as _provider_from_env,
    request_extra_from_env,
)
from simple_agent_lab.messages import AssistantMessage, message_text
from simple_agent_lab.state import State
from simple_agent_lab.trace import run_trace_from_state, write_event_stream


DEFAULT_TRACE_PATH = "/logs/agent/sal-trajectory.jsonl"
DEFAULT_SUMMARY_PATH = "/logs/agent/sal-summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--agent-flavor",
        choices=SIMPLE_AGENT_FLAVORS,
        default="bash_task_read",
    )
    parser.add_argument("--provider", choices=["openai", "fake"], default="openai")
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        # env-ok: CLI default mirrors API_KIND for the Harbor in-container entrypoint
        default=os.environ.get(API_KIND_ENV, DEFAULT_API_KIND),
    )
    parser.add_argument("--trace-id", default="harbor.simple-agent-lab")
    parser.add_argument("--producer", default="harbor:simple-agent-lab")
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def load_instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        return Path(args.instruction_file).read_text(encoding="utf-8")
    if args.instruction:
        return args.instruction
    raise SystemExit("missing instruction: pass --instruction or --instruction-file")


def provider_from_runner_args(args: argparse.Namespace) -> Provider:
    if args.provider == "fake":
        return FAKE_PROVIDER
    return _provider_from_env(
        OPENAI_ENV,
        api_kind=args.api_kind,
        default_temperature=1.0,
        read_reasoning=True,
        label="the Harbor SAL runner provider",
    )


def build_runner_session(
    *,
    provider: Provider,
    cwd: str | Path,
    agent_flavor: str,
    max_turns: int,
    request_extra: dict[str, Any] | None = None,
):
    if agent_flavor == "bash":
        return agent_session(
            provider,
            cwd=cwd,
            bash=True,
            read=False,
            general_purpose=False,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    if agent_flavor == "bash_task":
        return agent_session(
            provider,
            cwd=cwd,
            bash=True,
            read=False,
            general_purpose=True,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    if agent_flavor == "bash_task_read":
        return agent_session(
            provider,
            cwd=cwd,
            bash=True,
            read=True,
            general_purpose=True,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    if agent_flavor == "bash_skills":
        return agent_session(
            provider,
            cwd=cwd,
            bash=True,
            read=True,
            general_purpose=True,
            skills=True,
            request_extra=request_extra,
            max_turns=max_turns,
        )
    raise SystemExit(
        f"Unsupported --agent-flavor {agent_flavor!r}; expected {SIMPLE_AGENT_FLAVORS}."
    )


def final_text_from_state(state: State) -> str:
    for message in reversed(state.messages):
        if isinstance(message, AssistantMessage):
            text = message_text(message)
            if text:
                return text
    return ""


def _write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    instruction = load_instruction(args)
    provider = provider_from_runner_args(args)
    started = time.monotonic()
    request_extra = request_extra_from_env() if args.provider == "openai" else {}

    with build_runner_session(
        provider=provider,
        cwd=args.cwd,
        agent_flavor=args.agent_flavor,
        max_turns=args.max_turns,
        request_extra=request_extra,
    ) as session:
        state, events = session.run(instruction, max_turns=args.max_turns)
        for _event in events:
            pass
        try:
            trace_state = session.agent.trace_state(state)
        except Exception:
            trace_state = state

    trace = run_trace_from_state(
        state=trace_state,
        trace_id=args.trace_id,
        producer=args.producer,
        meta={
            "harbor": True,
            "agent_flavor": args.agent_flavor,
            "provider": args.provider,
            "cwd": str(args.cwd),
        },
    )
    write_event_stream(args.trace_path, trace)

    summary = {
        "status": "ok",
        "provider": args.provider,
        "model": provider.model,
        "agent_flavor": args.agent_flavor,
        "max_turns": args.max_turns,
        "cwd": str(args.cwd),
        "trace_path": str(args.trace_path),
        "final": final_text_from_state(state),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    _write_summary(args.summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_once(args)
    except BaseException as exc:
        summary = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "provider": getattr(args, "provider", None),
            "agent_flavor": getattr(args, "agent_flavor", None),
        }
        try:
            _write_summary(args.summary_path, summary)
        except Exception:
            pass
        print(
            f"[simple-agent-lab] runner failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
