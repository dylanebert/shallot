import { existsSync, readFileSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";
import { ENDPOINT_FILE } from "./playwright.global-setup";

// S4's touch verification gate (spec: `shallot-mobile-controls`) — a driver-level Playwright gate over
// gym's own `orbit-touch` scenario (`src/scenarios/orbit-touch.ts`), real CDP touch dispatch
// (`test/touch-dispatch.ts`) reading the orbit pose back through `window.__orbitPose()`. Same shape as
// `examples/showcase/roads/playwright.config.ts`: `shallot-mobile-controls` spec's Locked decision
// picked CDP `Input.dispatchTouchEvent` over `page.touchscreen`/synthetic `dispatchEvent` as the
// integration-honest instrument, Chromium-only by construction (`touch-dispatch.ts`'s header). WSL
// routes through `scripts/wsl-bridge.ts`'s host-GPU bridge via `playwright.global-setup.ts`, same as
// roads; off WSL (or when the bridge's prerequisites are absent) this falls through to a local launch,
// display-gated by the adapter-name skip in `test/touch.playwright.ts`.
//
// `hasTouch: true` plus a mobile device preset (`devices["Pixel 5"]`) is the context extension this
// gate needs — declared here rather than touching `bin/verify.ts`'s `browser.newContext()` call sites,
// which this gate never goes through (it drives Playwright's own `@playwright/test` runner directly
// against gym's dev server, an external driver in the same role `bun bench` or a human plays — the
// scenario's own `params`/`assert` stay environment-unaware, unchanged).
//
// Run by path — `cd examples/gym && bunx playwright test` (or `bun run gate`) — a by-path tier, never
// part of the default `bun run test` sweep.

const PORT = 3210;
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
