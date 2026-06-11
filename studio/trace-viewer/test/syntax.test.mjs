// Guards that index.html's inline script stays syntactically valid. The viewer
// is a single self-contained file with no build step, so a stray syntax error
// would only surface as a blank page in a browser. Parse the main <script>
// here (compile without executing) so it fails in CI instead.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { test } from "node:test";

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, "..", "index.html"), "utf8");

test("the main inline <script> compiles", () => {
  // Every <script> that is not the embedded JSON sample.
  const blocks = [
    ...html.matchAll(/<script(?![^>]*application\/json)[^>]*>([\s\S]*?)<\/script>/g),
  ].map((m) => m[1]);
  assert.ok(blocks.length >= 1, "no executable <script> block found");
  const js = blocks.join("\n;\n");
  // new Function compiles (syntax-checks) without running the body.
  assert.doesNotThrow(() => new Function(js), "index.html inline script has a syntax error");
});
