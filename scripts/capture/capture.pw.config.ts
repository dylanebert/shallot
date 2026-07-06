import { defineConfig } from "@playwright/test";

// Layered timeouts so a stuck flow fails in seconds, never hangs (a missing locator, a blank canvas).
// A single action/navigation caps at 10s; a whole flow at 15s (zoo-sweep scales its own budget per
// specimen) — Playwright enforces each and tears down its browser cleanly. The per-flow ceilings are
// the primary guard; `globalTimeout` only has to fit their worst-case sum across the full 26-flow
// suite (~900s). The harness adds a spawn ceiling above `globalTimeout` as a last-resort backstop
// (harness/core/playwright.ts), so the only thing past these is a wedged process.
export default defineConfig({
    testDir: "./flows",
    testMatch: "*.pw.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 15_000,
    globalTimeout: 900_000,

    expect: { timeout: 5_000 },

    use: {
        trace: "off",
        video: "off",
        headless: false,
        viewport: { width: 1920, height: 1080 },
        actionTimeout: 10_000,
        navigationTimeout: 10_000,
    },

    projects: [
        {
            name: "chromium",
            use: { channel: "chrome" },
        },
    ],
});
