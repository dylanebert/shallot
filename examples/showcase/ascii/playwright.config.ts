import { defineConfig } from "@playwright/test";

// The ascii showcase's own browser driver — bring-your-own, as a real user would (shallot exports no
// Playwright harness). `test/pixel-probe.playwright.ts` is this project's whole driver: criterion 5's
// masked pixel probe over the compositor (`shallot-tui` spec). The web server is `shallot dev` (the
// standalone runtime, no editor), so the gate runs against the same path a user opens. This is full
// device testing: it needs a capable WebGPU GPU, so the launch is local and headed on the
// session's display — `playwright.global-setup.ts` refuses to start without one, and the adapter-name
// skip in `test/gpu-adapter.ts` is the second guard against a software adapter.

const PORT = 3104;
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
        // standalone `shallot dev` over this project's manifest — `bunx` resolves the installed CLI. A cold
        // first vite build can run past 60s in CI; the warm cache serves in ~1s.
        command: `bunx shallot dev . --port ${PORT} --strict-port`,
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
            ],
        },
    },
});
