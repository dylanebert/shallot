import { defineConfig } from "@playwright/test";

// This project's own browser driver — bring-your-own, as a real user would. It needs a capable WebGPU
// GPU, so the launch is local and headed on the session's display — `playwright.global-setup.ts` refuses
// to start without one. `test/touch-smoke.playwright.ts` display-gates itself with an adapter-name skip;
// `test/visualization.playwright.ts` has none of its own and simply inherits whichever adapter this
// config resolves.

const PORT = 3118;
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
        command: `bun run build && bunx vite preview --port ${PORT} --strictPort`,
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
