/** Resolve a Chrome/Chromium binary for headless puppeteer-core smoke tests. */
import { accessSync, constants } from "node:fs";

const CANDIDATES = [
  process.env.CHROME_BIN,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);

export function resolveChromeExecutable() {
  for (const candidate of CANDIDATES) {
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      /* try next */
    }
  }
  throw new Error(
    "No Chrome/Chromium executable found. Set CHROME_BIN or install google-chrome / chromium.",
  );
}

export function launchArgs() {
  return ["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"];
}
