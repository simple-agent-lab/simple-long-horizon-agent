#!/usr/bin/env node
/**
 * Headless TTI profile for Observatory trace viewer.
 * Usage: node studio/trace-viewer/profile-trace.mjs [absolute-trace-path]
 */
import puppeteer from "puppeteer-core";

const TRACE_PATH =
  process.argv[2] ||
  "/Users/bytedance/projects/simple-agent-lab/evals/out/swebench_container_runs/swebench-first5-newprompt-20260527-170331/astropy__astropy-12907/out/trajectory.jsonl";
const BASE = process.env.TRACE_VIEWER_URL || "http://127.0.0.1:8765";
const CHROME =
  process.env.CHROME_BIN ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const url = `${BASE}/?load=${encodeURIComponent(TRACE_PATH)}&perf=1`;

async function main() {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--disable-gpu", "--no-sandbox"],
  });
  const page = await browser.newPage();
  const t0 = Date.now();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await page.waitForFunction(
    () => window.__perfReport?.ready === true,
    { timeout: 120_000 },
  );
  const wallMs = Date.now() - t0;
  const report = await page.evaluate(() => window.__perfReport);
  const rows = await page.evaluate(() => ({
    streamRows: document.querySelectorAll("#stream .row").length,
    treeNodes: document.querySelectorAll("#tree .node").length,
  }));
  await browser.close();
  console.log(
    JSON.stringify({ wallMs, rows, ...report }, null, 2),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
