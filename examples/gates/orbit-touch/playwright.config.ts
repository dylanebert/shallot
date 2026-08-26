import { existsSync, readFileSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";
import { ENDPOINT_FILE } from "./playwright.global-setup";

// S4's touch verification gate — real CDP touch dispatch (`touch-dispatch.ts`) against the fixture app
// in `src/main.ts`, reading the orbit pose back through `window.__orbitPose()`. Same shape as
// `examples/showcase/roads/playwright.config.ts`: `shallot-mobile-controls` spec's Locked decision
// picked CDP `Input.dispatchTouchEvent` over `page.touchscreen`/synthetic `dispatchEvent` as the
// integration-honest instrument, Chromium-only by construction (`touch-dispatch.ts`'s header) — the
// verify harness this gate extends is already Chromium-only, so no coverage is given up. WSL routes
// through `scripts/wsl-bridge.ts`'s host-GPU bridge via `playwright.global-setup.ts`, same as roads;
// off WSL (or when the bridge's prerequisites are absent) this falls through to a local launch,
// display-gated by the adapter-name skip in `test/touch.playwright.ts`.
//
// `hasTouch: true` plus a mobile device preset (`devices["Pixel 5"]`) is the one context extension this
// gate needs — none of the three existing showcase playwright configs set either today (spec's Locked
// decision), so it's declared here rather than touching `bin/verify.ts`'s `browser.newContext()` call
// sites, which this gate never goes through (it drives Playwright's own `@playwright/test` runner, not
// the `shallot verify` CLI).

const PORT = 3200;
const URL = `http://localhost:${PORT}`;

const endpoint = existsSync(ENDPOINT_FILE)
    ? (JSON.parse(readFileSync(ENDPOINT_FILE, "utf8")) as { wsEndpoint: string })
    : null;

export default defineConfig({
    testDir: "./test",
    testMatch: "*.playwright.ts",
    fullyParallel: false,
    workers: 1,
    reporter: [["list"]],
    timeout: 120_000,
    globalSetup: "./playwright.global-setup.ts",
    webServer: {
        command: `bunx vite --port ${PORT} --strictPort`,
        url: URL,
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
    },
    use: {
        ...devices["Pixel 5"],
        hasTouch: true,
        baseURL: URL,
        channel: "chrome",
        launchOptions: {
            args: [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-dawn-features=allow_unsafe_apis",
            ],
        },
        ...(endpoint ? { connectOptions: { wsEndpoint: endpoint.wsEndpoint } } : {}),
    },
});
