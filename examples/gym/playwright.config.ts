import { defineConfig, devices } from "@playwright/test";

// S4's touch verification gate (spec: `shallot-mobile-controls`) — a driver-level Playwright gate over
// gym's own `orbit-touch` scenario (`src/scenarios/orbit-touch.ts`), real CDP touch dispatch
// (`test/touch-dispatch.ts`) reading the orbit pose back through `window.__orbitPose()`. Same shape as
// `examples/showcase/roads/playwright.config.ts`: `shallot-mobile-controls` spec's Locked decision
// picked CDP `Input.dispatchTouchEvent` over `page.touchscreen`/synthetic `dispatchEvent` as the
// integration-honest instrument, Chromium-only by construction (`touch-dispatch.ts`'s header). The
// launch is local and headed on the session's display — `playwright.global-setup.ts` refuses to start
// without one, and the adapter-name skip in `test/touch.playwright.ts` is the second guard against a
// software adapter.
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
        // headed always. Measured on this seat (Omarchy/Hyprland, RTX 4090, driver 610.57.04,
        // 2026-09-08): headed system Chrome over a localhost origin reports `nvidia / lovelace`, while
        // headless reports `google / swiftshader` — or no adapter at all — under every flag set tried.
        // The channel does not avoid the software fallback headless. A window appears on the session's
        // display during a run; `playwright.global-setup.ts` refuses to start when there is no display,
        // so this never silently degrades to software.
        headless: false,
        ...devices["Pixel 5"],
        hasTouch: true,
        baseURL: URL,
        channel: "chrome",
        launchOptions: {
            args: [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-dawn-features=allow_unsafe_apis",
                // a headed gate window, placed out of the way by the session's own compositor rule
                "--class=kex-gate",
            ],
        },
    },
});
