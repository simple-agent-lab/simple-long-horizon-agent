#!/usr/bin/env node
/**
 * Entry point for the simple-agent-lab terminal UI.
 *
 * Constructs the pi-tui stack (`ProcessTerminal` + `TUI`), spawns the Python
 * gateway, and hands both to {@link App}. By default the gateway is launched
 * via `uv run python -m simple_agent_lab.tui_gateway.entry` so it inherits the
 * project's virtualenv; set `SAL_PYTHON` to point at an explicit interpreter
 * instead.
 */

import { resolve } from "node:path";
import { ProcessTerminal, TUI } from "@earendil-works/pi-tui";
import { App } from "./app.js";
import { GatewayClient } from "./gatewayClient.js";

interface Args {
  provider: string;
  maxTurns: number;
  showThinking: boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { provider: "fake", maxTurns: 12, showThinking: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--provider") args.provider = argv[++i] ?? args.provider;
    else if (a === "--max-turns") args.maxTurns = Number(argv[++i]) || args.maxTurns;
    else if (a === "--no-thinking") args.showThinking = false;
  }
  return args;
}

function gatewayLaunch(): { command: string; args: string[] } {
  const moduleArgs = ["-m", "simple_agent_lab.tui_gateway.entry"];
  const explicit = process.env.SAL_PYTHON;
  if (explicit) return { command: explicit, args: moduleArgs };
  // Default: go through uv so the project venv (anthropic/openai) is present.
  return { command: "uv", args: ["run", "python", ...moduleArgs] };
}

function main(): void {
  if (!process.stdin.isTTY) {
    process.stderr.write("sal-tui needs an interactive terminal (stdin is not a TTY).\n");
    process.exit(1);
  }

  const args = parseArgs(process.argv.slice(2));
  // src/entry.ts and dist/entry.js are both one level under studio/tui, so the
  // repo root is three levels up from this file in either case.
  const repoRoot = resolve(import.meta.dirname, "../../..");

  const launch = gatewayLaunch();
  const client = new GatewayClient({ command: launch.command, args: launch.args, cwd: repoRoot });

  const terminal = new ProcessTerminal();
  const tui = new TUI(terminal);

  const app = new App(tui, client, {
    provider: args.provider,
    cwd: repoRoot,
    maxTurns: args.maxTurns,
    showThinking: args.showThinking,
  });

  void app.start();
}

main();
