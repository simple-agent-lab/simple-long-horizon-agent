# simple-agent-lab TUI

A terminal UI for the agent runtime, built on
[`@earendil-works/pi-tui`](https://www.npmjs.com/package/@earendil-works/pi-tui).

It follows the Hermes-style split: this TypeScript process owns **only**
rendering and input, and talks to a Python **gateway**
(`simple_agent_lab.tui_gateway`) over newline-delimited JSON-RPC on the
gateway's stdio. The language boundary is what lets us reuse pi's polished
components (Markdown, Editor, Loader, …) while the agent stays pure Python.

```
┌── studio/tui (this) ──┐  stdio JSON-RPC  ┌── simple_agent_lab.tui_gateway ──┐
│ ProcessTerminal + TUI │ ───────────────► │ session.create / prompt.submit   │
│ Editor / Markdown /   │ ◄─────────────── │ core.run(...) events → wire      │
│ Loader  (pi-tui)      │   events         │ (Python agent + tools)           │
└───────────────────────┘                  └──────────────────────────────────┘
```

## Run

```bash
cd studio/tui
npm install

# dev (no build step; tsx runs the TS directly)
npm run dev -- --provider fake        # deterministic, no credentials
npm run dev -- --provider openai      # live model; reads .env from repo root

# or build a single-file bundle and run that
npm run build && node dist/entry.js --provider fake
```

By default the gateway is spawned via `uv run python -m
simple_agent_lab.tui_gateway.entry` so it inherits the project venv. Override
the interpreter with `SAL_PYTHON=/path/to/python`.

Keys: **Enter** send · **Esc** interrupt a running turn · **Ctrl-C** quit.

## Files

- `src/entry.ts` — builds `ProcessTerminal` + `TUI`, spawns the gateway, starts `App`.
- `src/gatewayClient.ts` — spawns the gateway and frames JSON-RPC; a single
  read loop dispatches responses by id and events to listeners (so the
  `prompt.submit` ack can never race a turn event into a deadlock).
- `src/app.ts` — maps gateway events onto pi-tui components.
- `src/theme.ts` — chalk-based theme objects for the themed components.

The Python REPL at `scripts/tui_gateway_repl.py` is a text-only stand-in for
this UI — handy for exercising the protocol without Node.
