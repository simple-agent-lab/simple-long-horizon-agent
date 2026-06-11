// Consumer-side contract test (Defense 2).
//
// The viewer's schema accessors live in ONE extractable block in index.html
// (between the `BEGIN/END trace-schema accessors` sentinels). This test pulls
// that EXACT block out, evaluates it, and runs it against the real generated
// fixture (sample-trace.jsonl, produced by tests/unit/test_trace_fixture_golden.py
// from the real Python serializer). So:
//   - if the runtime renames a trace field, the regenerated fixture changes and
//     these accessors return null on it -> this test goes red;
//   - if someone edits the accessors in index.html, this test runs the EDITED
//     code (no mirror copy to drift).
// That closes the producer<->viewer gap the `data`->`sidecar` bug fell through.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { test } from "node:test";

const HERE = dirname(fileURLToPath(import.meta.url));
const INDEX_HTML = join(HERE, "..", "index.html");
const SAMPLE = join(HERE, "..", "sample-trace.jsonl");

const BEGIN = "// ===== BEGIN trace-schema accessors =====";
const END = "// ===== END trace-schema accessors =====";

function loadTraceSchema() {
  const html = readFileSync(INDEX_HTML, "utf8");
  const start = html.indexOf(BEGIN);
  const end = html.indexOf(END);
  assert.ok(start !== -1 && end !== -1 && end > start, "trace-schema block not found in index.html");
  const block = html.slice(start, end);
  // The block defines `const TraceSchema = (...)();`; hand it back out.
  return new Function(`${block}\n;return TraceSchema;`)();
}

function loadFixture() {
  return JSON.parse(readFileSync(SAMPLE, "utf8").trim());
}

const TraceSchema = loadTraceSchema();
const fixture = loadFixture();
const messages = fixture.events.filter((e) => e.kind === "message").map((e) => e.message);

test("supported schema matches the fixture", () => {
  assert.equal(TraceSchema.SUPPORTED_SCHEMA, fixture.schema);
});

test("wireRaw resolves the provider raw blob (Wire debug panel)", () => {
  const withRaw = messages.find((m) => TraceSchema.wireRaw(m));
  assert.ok(withRaw, "no message exposed a raw blob through the accessor");
  const raw = TraceSchema.wireRaw(withRaw);
  assert.ok(raw.request && raw.response, "raw blob missing request/response");
  // The accessor must surface the reasoning field the panel shows.
  assert.equal(raw.request.reasoning_effort, "high");
});

test("modelName resolves from the per-call raw blob", () => {
  const withRaw = messages.find((m) => TraceSchema.wireRaw(m));
  const name = TraceSchema.modelName(withRaw, fixture.meta);
  assert.ok(name && typeof name === "string", "model name should be non-empty");
  assert.ok(!name.includes("/"), "provider/ prefix should be stripped");
});

test("sidecarDetails + subEvents resolve the sub-agent drill-down", () => {
  const withDetails = messages.find((m) => TraceSchema.sidecarDetails(m));
  assert.ok(withDetails, "no message exposed sub-agent details through the accessor");
  const details = TraceSchema.sidecarDetails(withDetails);
  const callId = Object.keys(details)[0];
  const sub = TraceSchema.subEvents(withDetails, callId);
  assert.ok(Array.isArray(sub) && sub.length > 0, "sub_events should be a non-empty array");
  assert.ok(sub.some((e) => e.kind === "model_request"), "sub-agent trace should contain events");
});

test("accessors go null when the producer renames the sidecar slot (drift tripwire)", () => {
  // Simulate the historical bug in reverse: rename the slot the producer emits.
  const renamed = JSON.parse(JSON.stringify(fixture), (k, v) => v);
  const drift = (o) => {
    if (Array.isArray(o)) o.forEach(drift);
    else if (o && typeof o === "object") {
      if ("sidecar" in o) { o.data_renamed = o.sidecar; delete o.sidecar; }
      Object.values(o).forEach(drift);
    }
  };
  drift(renamed);
  const msgs = renamed.events.filter((e) => e.kind === "message").map((e) => e.message);
  assert.equal(msgs.find((m) => TraceSchema.wireRaw(m)), undefined,
    "accessor must fail to find raw once the producer field is renamed");
});
