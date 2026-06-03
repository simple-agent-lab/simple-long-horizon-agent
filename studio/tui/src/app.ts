/**
 * App — maps gateway events onto pi-tui components.
 *
 * Layout (top to bottom): a growing `transcript` Container of past
 * messages/tool calls, a turn-level status `Loader`, and the focused `Editor`
 * for input. Each event appends or mutates real pi-tui components and asks the
 * TUI to re-render — this is where we actually reuse pi's rendering layer
 * (Markdown for assistant prose, Loader spinners for running tools, Editor for
 * input) instead of hand-printing ANSI.
 */

import {
  type Component,
  Container,
  Editor,
  isKeyRelease,
  isKeyRepeat,
  Key,
  Loader,
  matchesKey,
  Spacer,
  Text,
  type TUI,
} from "@earendil-works/pi-tui";
import { type GatewayClient, type GatewayEvent, type RpcError } from "./gatewayClient.js";
import { SpacedMarkdown } from "./spacedMarkdown.js";
import { c, editorTheme, markdownTheme } from "./theme.js";
import { ToolCallBox } from "./toolCall.js";

export interface AppOptions {
  provider: string;
  cwd: string;
  maxTurns: number;
  showThinking: boolean;
}

export class App {
  private transcript = new Container();
  private statusContainer = new Container();
  private editorContainer = new Container();
  private statusLoader: Loader;
  private editor: Editor;
  private toolBoxes = new Map<string, ToolCallBox>();
  // Mirrors pi coding-agent: Ctrl-O toggles a global tool expansion mode, and
  // each existing/future tool box reads that mode.
  private allToolBoxes: ToolCallBox[] = [];
  private toolOutputExpanded = false;
  private sessionId = "";
  private busy = false;

  constructor(
    private tui: TUI,
    private client: GatewayClient,
    private opts: AppOptions,
  ) {
    this.statusLoader = new Loader(tui, c.accent, c.muted, "");
    this.editor = new Editor(tui, editorTheme, { paddingX: 1 });
    this.editor.onSubmit = (text) => this.submit(text);

    tui.addChild(this.transcript);
    this.statusContainer.addChild(this.statusLoader);
    this.editorContainer.addChild(this.editor);
    tui.addChild(this.statusContainer);
    tui.addChild(this.editorContainer);

    // pi-tui's Loader auto-starts its spinner in the constructor, so the idle
    // status line would spin (and repaint every 80ms) forever. Park it.
    this.stopStatus();

    client.onEvent((event) => this.onEvent(event));
    client.onStderr((line) => this.onStderr(line));
    client.onExit(() => this.shutdown(0));
    tui.addInputListener((data) => this.onGlobalKey(data));
  }

  async start(): Promise<void> {
    this.tui.start();
    this.tui.setFocus(this.editor);
    try {
      const info = await this.client.request<{ session_id: string; info: Record<string, unknown> }>(
        "session.create",
        { provider: this.opts.provider, cwd: this.opts.cwd },
      );
      this.sessionId = info.session_id;
      const meta = info.info;
      this.append(
        new Text(
          c.muted(
            `session ${info.session_id} · model=${meta.model} · tools=${JSON.stringify(meta.tools)}`,
          ),
          1,
          0,
        ),
      );
      this.append(
        new Text(
          c.muted("Enter send · Esc interrupt · Ctrl-O expand output · Ctrl-C quit"),
          1,
          0,
        ),
      );
      this.append(new Spacer(1));
      this.tui.requestRender();
    } catch (err) {
      this.fatal(`session.create failed: ${this.errText(err)}`);
    }
  }

  // -- input -------------------------------------------------------------

  private submit(text: string): void {
    const trimmed = text.trim();
    if (!trimmed || this.busy) return;
    this.editor.setText("");
    this.editor.addToHistory(trimmed);
    this.appendBlock(new Text(`${c.user("you ›")} ${trimmed}`, 1, 0));

    this.busy = true;
    this.editor.disableSubmit = true;
    this.startStatus(c.muted("working…"));
    this.tui.requestRender();

    this.client
      .request("prompt.submit", {
        session_id: this.sessionId,
        text: trimmed,
        max_turns: this.opts.maxTurns,
      })
      .catch((err) => this.showError(this.errText(err)));
  }

  private onGlobalKey(data: string): { consume?: boolean } | undefined {
    if (matchesKey(data, Key.ctrl("o"))) {
      if (!isKeyRelease(data) && !isKeyRepeat(data)) {
        this.setToolsExpanded(!this.toolOutputExpanded);
      }
      return { consume: true };
    }
    if (matchesKey(data, Key.ctrl("c"))) {
      if (this.busy) {
        this.client.request("session.interrupt", { session_id: this.sessionId }).catch(() => {});
      } else {
        this.shutdown(0);
      }
      return { consume: true };
    }
    if (this.busy && matchesKey(data, Key.escape)) {
      this.client.request("session.interrupt", { session_id: this.sessionId }).catch(() => {});
      return { consume: true };
    }
    return undefined;
  }

  // -- events ------------------------------------------------------------

  private onEvent(event: GatewayEvent): void {
    const p = event.payload;
    switch (event.type) {
      case "thinking":
        if (this.opts.showThinking) {
          this.appendBlock(new Text(c.dim(`  ⋮ ${this.oneLine(String(p.text ?? ""), 140)}`), 1, 0));
        }
        break;
      case "message.complete":
        // Only the final answer is rendered. The intermediate "step" preamble
        // ("好的，我来跑…") is dropped — it duplicates the assistant label and
        // the tool box already shows what's happening.
        if (p.is_final) {
          this.append(new Text(c.assistant("assistant ›"), 1, 0));
          this.appendBlock(new SpacedMarkdown(String(p.text ?? ""), 1, 0, markdownTheme));
        }
        break;
      case "tool.start":
        this.onToolStart(event);
        break;
      case "tool.complete":
        this.onToolComplete(event);
        break;
      case "status.update":
        this.append(new Text(c.warn(`  [${p.kind}] ${p.before_tokens}→${p.after_tokens} tokens`), 1, 0));
        break;
      case "turn.complete":
        this.endTurn();
        break;
      case "error":
        this.showError(String(p.message ?? "unknown error"));
        break;
      // session.info / message.start: no view surface here.
    }
  }

  private onToolStart(event: GatewayEvent): void {
    const p = event.payload;
    const id = String(p.tool_call_id ?? "");
    const title = String(p.title || p.name || "tool");
    const description = p.description ? String(p.description) : "";
    const box = new ToolCallBox(this.tui, title, description);
    box.setExpanded(this.toolOutputExpanded);
    this.toolBoxes.set(id, box);
    this.allToolBoxes.push(box);
    this.append(box);
  }

  private onToolComplete(event: GatewayEvent): void {
    const p = event.payload;
    const id = String(p.tool_call_id ?? "");
    const isError = Boolean(p.is_error);
    const view = {
      state: isError ? ("error" as const) : ("success" as const),
      allLines: this.cleanToolText(String(p.text ?? "")).split("\n"),
      durationS: typeof p.duration_s === "number" ? p.duration_s : undefined,
    };

    const box = this.toolBoxes.get(id);
    if (box) {
      box.setResult(view);
      this.toolBoxes.delete(id);
    } else {
      // No matching start (shouldn't happen) — synthesize a finished box.
      const fresh = new ToolCallBox(this.tui, String(p.title || p.name || "tool"), "");
      fresh.setExpanded(this.toolOutputExpanded);
      fresh.setResult(view);
      this.allToolBoxes.push(fresh);
      this.append(fresh);
    }
    this.append(new Spacer(1)); // breathing room after the tool box
    this.tui.requestRender();
  }

  // -- helpers -----------------------------------------------------------

  private endTurn(): void {
    // Any tool still "running" (e.g. on interrupt) stops animating.
    for (const box of this.toolBoxes.values()) box.cancel();
    this.toolBoxes.clear();
    this.busy = false;
    this.editor.disableSubmit = false;
    this.stopStatus();
    this.tui.setFocus(this.editor);
    this.tui.requestRender();
  }

  /** Spin the status line with `msg` (default braille frames). */
  private startStatus(msg: string): void {
    this.statusLoader.setIndicator(undefined); // default frames; starts spinner
    this.statusLoader.setMessage(msg);
  }

  /** Stop and blank the status line — no interval, no spinner glyph. */
  private stopStatus(): void {
    this.statusLoader.stop();
    this.statusLoader.setIndicator({ frames: [] }); // no glyph
    this.statusLoader.setMessage("");
  }

  private showError(message: string): void {
    this.append(new Text(c.error(`  ✗ ${message}`), 1, 0));
    this.endTurn();
  }

  private append(component: Component): void {
    this.transcript.addChild(component);
    this.tui.requestRender();
  }

  /** Append a block followed by a blank line, for even vertical rhythm. */
  private appendBlock(component: Component): void {
    this.transcript.addChild(component);
    this.transcript.addChild(new Spacer(1));
    this.tui.requestRender();
  }

  private setToolsExpanded(expanded: boolean): void {
    this.toolOutputExpanded = expanded;
    for (const box of this.allToolBoxes) {
      box.setExpanded(expanded);
    }
    this.tui.requestRender();
  }

  private onStderr(line: string): void {
    // The gateway's log channel. Keep it quiet unless it looks like a traceback.
    if (/error|traceback|exception/i.test(line)) {
      this.append(new Text(c.dim(`  · ${line}`), 1, 0));
    }
  }

  /** Drop the bash result's redundant leading "$ cmd" and "stdout:" label. */
  private cleanToolText(text: string): string {
    const lines = text.split("\n");
    if (lines[0]?.startsWith("$ ")) lines.shift();
    return lines.filter((l) => l !== "stdout:").join("\n");
  }

  private oneLine(text: string, limit: number): string {
    const flat = text.replace(/\s+/g, " ").trim();
    return flat.length > limit ? `${flat.slice(0, limit - 1)}…` : flat;
  }

  private errText(err: unknown): string {
    if (err && typeof err === "object" && "message" in err) {
      return String((err as RpcError).message);
    }
    return String(err);
  }

  private fatal(message: string): void {
    this.tui.stop();
    process.stderr.write(`${message}\n`);
    if (this.opts.provider === "openai") {
      process.stderr.write(
        "openai provider needs OPENAI_MODEL + OPENAI_AUTH_TOKEN (the gateway reads .env from cwd).\n",
      );
    }
    this.client.stop();
    process.exit(1);
  }

  private shutdown(code: number): void {
    this.tui.stop();
    this.client.stop();
    process.exit(code);
  }
}
