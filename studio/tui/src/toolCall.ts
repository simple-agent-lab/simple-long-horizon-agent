/**
 * ToolCallBox — a pi-coding-agent-style tool call rendered as a full-width
 * box whose BACKGROUND COLOR encodes state (pending / success / error),
 * rather than a leading status glyph.
 *
 * pi ships this look in its (unpublished) coding-agent app, not in the pi-tui
 * package, so we rebuild it here on the primitives: each line is padded to the
 * viewport width and wrapped in a chalk background function. While running, a
 * braille spinner + live elapsed timer animate in the footer; on completion
 * the background flips to green/red and the footer shows "Took Ns".
 *
 * Output is collapsed to PREVIEW_LINES with a "(+N more · ctrl+o)" hint; the
 * box keeps the FULL output and {@link setExpanded} switches between preview
 * and full, mirroring pi's expand affordance.
 *
 * Implements the pi-tui `Component` contract directly (render/invalidate) so
 * the TUI can place it in the transcript like any built-in component.
 */

import { type Component, truncateToWidth, type TUI, visibleWidth } from "@earendil-works/pi-tui";
import { c } from "./theme.js";

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const FRAME_MS = 80;
/** Output lines shown before collapsing (pi caps bash near 20). */
const PREVIEW_LINES = 16;
/** Left inset inside the box. */
const PAD = " ";

export type ToolState = "pending" | "success" | "error";

export interface ToolResultView {
  state: ToolState;
  /** The full tool output, already cleaned of redundant scaffolding. */
  allLines: string[];
  durationS?: number;
}

export class ToolCallBox implements Component {
  private state: ToolState = "pending";
  private allLines: string[] = [];
  private expanded = false;
  private durationS?: number;
  private frame = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private readonly startedAt = Date.now();

  constructor(
    private tui: TUI,
    private title: string,
    private description: string,
  ) {
    this.timer = setInterval(() => {
      this.frame = (this.frame + 1) % SPINNER.length;
      this.tui.requestRender();
    }, FRAME_MS);
  }

  /** Finalize the box with the tool's result and stop the animation. */
  setResult(view: ToolResultView): void {
    this.state = view.state;
    this.allLines = view.allLines;
    this.durationS = view.durationS;
    this.stopTimer();
    this.tui.requestRender();
  }

  /** Show full output (true) or the collapsed preview (false). */
  setExpanded(expanded: boolean): void {
    if (this.expanded === expanded) return;
    this.expanded = expanded;
    this.tui.requestRender();
  }

  /** Whether this box has more output than the collapsed preview shows. */
  get collapsible(): boolean {
    return this.allLines.length > PREVIEW_LINES;
  }

  /** Stop animating without a result (e.g. the turn was interrupted). */
  cancel(): void {
    if (this.state === "pending") this.state = "error";
    this.stopTimer();
  }

  invalidate(): void {}

  render(width: number): string[] {
    const bg = this.background();
    const out: string[] = [];
    out.push(this.fill("", width, bg)); // top padding row

    const header = this.description
      ? `${c.boxTitle(this.title)}  ${c.boxDesc(this.description)}`
      : c.boxTitle(this.title);
    out.push(this.fill(`${PAD}${header}`, width, bg));

    const limit = this.expanded ? this.allLines.length : PREVIEW_LINES;
    const shown = this.allLines.slice(0, limit);
    const bodyFn = this.state === "error" ? c.boxError : c.boxBody;
    for (const line of shown) {
      out.push(this.fill(`${PAD}  ${bodyFn(line)}`, width, bg));
    }

    const hidden = this.allLines.length - shown.length;
    if (hidden > 0) {
      const hint = `… (+${hidden} more line${hidden === 1 ? "" : "s"} · ctrl+o to expand)`;
      out.push(this.fill(`${PAD}  ${c.boxFooter(hint)}`, width, bg));
    } else if (this.expanded && this.allLines.length > PREVIEW_LINES) {
      out.push(this.fill(`${PAD}  ${c.boxFooter("ctrl+o to collapse")}`, width, bg));
    }

    out.push(this.fill(`${PAD}${c.boxFooter(this.footer())}`, width, bg));
    out.push(this.fill("", width, bg)); // bottom padding row
    return out;
  }

  private footer(): string {
    if (this.state === "pending") {
      const elapsed = (Date.now() - this.startedAt) / 1000;
      return `Running ${SPINNER[this.frame]} ${elapsed.toFixed(1)}s`;
    }
    const took = this.durationS != null ? ` ${this.durationS.toFixed(1)}s` : "";
    return this.state === "error" ? `Failed${took}` : `Took${took}`;
  }

  private background(): (s: string) => string {
    if (this.state === "pending") return c.bgPending;
    if (this.state === "error") return c.bgError;
    return c.bgSuccess;
  }

  /** Truncate content to the width, pad with spaces, and apply the bg color. */
  private fill(content: string, width: number, bg: (s: string) => string): string {
    const trimmed = truncateToWidth(content, width);
    const pad = Math.max(0, width - visibleWidth(trimmed));
    return bg(trimmed + " ".repeat(pad));
  }

  private stopTimer(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}
