/**
 * SpacedMarkdown — pi-tui's Markdown, but with breathing room between list
 * items.
 *
 * pi-tui's `renderList` emits list items back-to-back with no blank line
 * between them (it ignores CommonMark loose/tight lists), so a bullet list
 * reads denser than the surrounding paragraphs. We can't configure that, so we
 * subclass and post-process the rendered lines: insert one blank line before
 * each list-item line (when the previous line isn't already blank). Wrapped
 * continuation lines use a space prefix, not a bullet, so they stay attached to
 * their item — only the item boundaries get air.
 */

import { Markdown } from "@earendil-works/pi-tui";

const ESC = String.fromCharCode(27);
// Strip SGR color codes so the bullet test runs on plain text.
const ANSI = new RegExp(`${ESC}\\[[0-9;]*m`, "g");
// A rendered list item starts with an optional indent then a bullet or number.
const LIST_ITEM = /^\s*([-*+]|\d+\.)\s+\S/;

export class SpacedMarkdown extends Markdown {
  render(width: number): string[] {
    const lines = super.render(width);
    const out: string[] = [];
    for (const line of lines) {
      const plain = line.replace(ANSI, "");
      const prev = out.length ? out[out.length - 1].replace(ANSI, "") : "";
      if (LIST_ITEM.test(plain) && prev.trim() !== "") {
        out.push("");
      }
      out.push(line);
    }
    return out;
  }
}
