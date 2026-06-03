/**
 * Chalk-based theme objects for the pi-tui components.
 *
 * pi-tui ships no theme of its own — every themed component takes a bag of
 * `(text) => styledText` functions in its constructor. We build those here
 * with `chalk` so the look lives in one place, mirroring how the pi coding
 * agent keeps its colors in a theme module.
 */

import chalk from "chalk";
import type {
  EditorTheme,
  MarkdownTheme,
  SelectListTheme,
} from "@earendil-works/pi-tui";

/** Small palette reused across the app's own (non-component) rendering. */
export const c = {
  dim: (s: string) => chalk.dim(s),
  bold: (s: string) => chalk.bold(s),
  accent: (s: string) => chalk.cyan(s),
  user: (s: string) => chalk.green(s),
  assistant: (s: string) => chalk.green(s),
  tool: (s: string) => chalk.cyan(s),
  ok: (s: string) => chalk.green(s),
  error: (s: string) => chalk.red(s),
  warn: (s: string) => chalk.yellow(s),
  muted: (s: string) => chalk.gray(s),
  // Tool-call box backgrounds, encoding state the way the pi coding agent does:
  // pending = dark blue-gray, success = dark green, error = dark red.
  bgPending: (s: string) => chalk.bgHex("#282832")(s),
  bgSuccess: (s: string) => chalk.bgHex("#283228")(s),
  bgError: (s: string) => chalk.bgHex("#3c2828")(s),
  // Foreground styles used inside the colored boxes.
  boxTitle: (s: string) => chalk.bold.whiteBright(s),
  boxDesc: (s: string) => chalk.gray(s),
  boxBody: (s: string) => chalk.white(s),
  boxFooter: (s: string) => chalk.gray(s),
  boxError: (s: string) => chalk.redBright(s),
};

export const markdownTheme: MarkdownTheme = {
  heading: (t) => chalk.bold.cyan(t),
  link: (t) => chalk.cyan.underline(t),
  linkUrl: (t) => chalk.dim.cyan(t),
  code: (t) => chalk.yellow(t),
  codeBlock: (t) => chalk.gray(t),
  codeBlockBorder: (t) => chalk.dim(t),
  quote: (t) => chalk.italic.gray(t),
  quoteBorder: (t) => chalk.dim(t),
  hr: (t) => chalk.dim(t),
  listBullet: (t) => chalk.cyan(t),
  bold: (t) => chalk.bold(t),
  italic: (t) => chalk.italic(t),
  strikethrough: (t) => chalk.strikethrough(t),
  underline: (t) => chalk.underline(t),
  codeBlockIndent: "  ",
};

const selectListTheme: SelectListTheme = {
  selectedPrefix: (t) => chalk.cyan(t),
  selectedText: (t) => chalk.cyan.bold(t),
  description: (t) => chalk.dim(t),
  scrollInfo: (t) => chalk.dim(t),
  noMatch: (t) => chalk.dim(t),
};

export const editorTheme: EditorTheme = {
  borderColor: (t) => chalk.dim(t),
  selectList: selectListTheme,
};
