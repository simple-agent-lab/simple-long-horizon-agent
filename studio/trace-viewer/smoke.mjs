#!/usr/bin/env node
/**
 * Page-level RPA smoke tests for the Observatory trace viewer.
 *
 * Requires a running serve.py instance. The run script sets TRACE_VIEWER_URL.
 *
 *   node studio/trace-viewer/smoke.mjs
 */
import puppeteer from "puppeteer-core";
import { launchArgs, resolveChromeExecutable } from "./lib/chrome.mjs";

const BASE = (process.env.TRACE_VIEWER_URL || "http://127.0.0.1:8765").replace(
  /\/$/,
  "",
);
const SAMPLE_TRACE = "evals/out/_smoke/trajectory.jsonl";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function attachErrorCollectors(page, bucket) {
  page.on("pageerror", (err) => bucket.push(`pageerror: ${err}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (/favicon|Failed to load resource/i.test(text)) return;
    bucket.push(`console: ${text}`);
  });
  page.on("response", (response) => {
    const url = response.url();
    if (url.includes("favicon.ico")) return;
    if (response.status() >= 400) {
      bucket.push(`http ${response.status()} ${url}`);
    }
  });
}

async function openPage(browser) {
  const page = await browser.newPage();
  const errors = [];
  attachErrorCollectors(page, errors);
  return { page, errors };
}

async function waitForTree(page, minNodes = 5) {
  await page.waitForFunction(
    (n) => document.querySelectorAll("#tree .node").length >= n,
    { timeout: 60_000 },
    minNodes,
  );
}

async function waitForPerfReady(page) {
  await page.waitForFunction(() => window.__perfReport?.ready === true, {
    timeout: 60_000,
  });
}

async function readRenderStats(page) {
  return page.evaluate(() => ({
    treeNodes: document.querySelectorAll("#tree .node").length,
    streamRows: document.querySelectorAll("#stream .row").length,
    statStrip: document.querySelector("#stat-strip")?.textContent || "",
    title: document.title,
    hasNaN: /\bNaN\b/.test(document.body.innerText),
    perf: window.__perfReport,
  }));
}

const tests = [
  {
    name: "embedded sample renders structure and stream",
    async run({ page, errors }) {
      await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await waitForTree(page, 20);
      const stats = await readRenderStats(page);
      assert(stats.treeNodes >= 20, `expected >= 20 tree nodes, got ${stats.treeNodes}`);
      assert(stats.streamRows >= 5, `expected >= 5 stream rows, got ${stats.streamRows}`);
      assert(
        stats.title.includes("Observatory"),
        `unexpected document title: ${stats.title}`,
      );
      assert(!stats.hasNaN, "page text contains NaN");
      assert(errors.length === 0, `unexpected page errors: ${errors.join("; ")}`);
    },
  },
  {
    name: "api load renders sample trace with perf markers",
    async run({ page, errors }) {
      const url = `${BASE}/?load=${encodeURIComponent(SAMPLE_TRACE)}&perf=1`;
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await waitForPerfReady(page);
      const stats = await readRenderStats(page);
      assert(stats.perf?.ready === true, "perf report not ready");
      assert(
        (stats.perf?.eventCount ?? 0) >= 40,
        `expected >= 40 events, got ${stats.perf?.eventCount}`,
      );
      assert(stats.treeNodes >= 20, `expected >= 20 tree nodes, got ${stats.treeNodes}`);
      assert(stats.streamRows >= 5, `expected >= 5 stream rows, got ${stats.streamRows}`);
      assert(/errors/i.test(stats.statStrip), "stat strip missing error count");
      assert(/compressions/i.test(stats.statStrip), "stat strip missing compression count");
      assert(/final/i.test(stats.statStrip), "stat strip missing exit reason");
      assert(!stats.hasNaN, "page text contains NaN");
      assert(errors.length === 0, `unexpected page errors: ${errors.join("; ")}`);
    },
  },
  {
    name: "view mode toggles change the active stream mode",
    async run({ page }) {
      await page.goto(`${BASE}/?load=${encodeURIComponent(SAMPLE_TRACE)}&perf=1`, {
        waitUntil: "domcontentloaded",
        timeout: 60_000,
      });
      await waitForPerfReady(page);

      const modes = ["messages", "events", "model"];
      const seen = new Set();
      for (const mode of modes) {
        const btn = await page.$(`[data-mode="${mode}"]`);
        assert(btn, `missing view toggle for ${mode}`);
        await btn.click();
        await sleep(120);
        const active = await page.$eval("[data-mode].active", (el) =>
          el.getAttribute("data-mode"),
        );
        assert(active === mode, `expected active view ${mode}, got ${active}`);
        seen.add(active);
      }
      assert(seen.size === 3, `expected 3 distinct active views, got ${[...seen]}`);
    },
  },
  {
    name: "tree selection populates the inspector",
    async run({ page }) {
      await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await waitForTree(page, 10);
      await page.click("#tree .node");
      await sleep(150);
      const inspectorLen = await page.$eval(
        "#inspector",
        (el) => (el.textContent || "").trim().length,
      );
      assert(
        inspectorLen > 40,
        `inspector too short after tree click (${inspectorLen} chars)`,
      );
    },
  },
  {
    name: "experiments sidebar loads a scanned trajectory",
    async run({ page, errors }) {
      await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await waitForTree(page, 10);
      await page.waitForFunction(
        () => document.querySelectorAll("#exp-list .exp-item").length >= 1,
        { timeout: 60_000 },
      );
      await page.click("#exp-list .exp-item");
      await waitForTree(page, 5);
      const stats = await readRenderStats(page);
      assert(stats.treeNodes >= 5, "sidebar load did not render a trace tree");
      assert(!stats.hasNaN, "page text contains NaN after sidebar load");
      assert(errors.length === 0, `unexpected page errors: ${errors.join("; ")}`);
    },
  },
];

async function main() {
  const chrome = resolveChromeExecutable();
  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: launchArgs(),
  });

  const failures = [];
  try {
    for (const test of tests) {
      const { page, errors } = await openPage(browser);
      try {
        process.stdout.write(`  · ${test.name} … `);
        await test.run({ page, errors, browser });
        console.log("ok");
      } catch (err) {
        console.log("FAIL");
        failures.push({ name: test.name, error: err });
      } finally {
        await page.close();
      }
    }
  } finally {
    await browser.close();
  }

  if (failures.length) {
    console.error(`\n${failures.length} trace viewer smoke test(s) failed:\n`);
    for (const { name, error } of failures) {
      console.error(`  ✗ ${name}`);
      console.error(`    ${error.message || error}`);
    }
    process.exit(1);
  }

  console.log(`\nAll ${tests.length} trace viewer smoke tests passed (${BASE}).`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
