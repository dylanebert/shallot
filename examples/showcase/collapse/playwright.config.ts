import { existsSync, readFileSync } from "node:fs";
import { defineConfig } from "@playwright/test";
import { ENDPOINT_FILE } from "./playwright.global-setup";

// The collapse showcase's own browser driver — bring-your-own, as a real user would. shallot exports no
// Playwright harness (bun ships `bun test` and tells you to bring Playwright); `test/touch-smoke.playwright.ts`
// (`shallot-mobile-controls` spec, S5) is this project's whole driver — a `hasTouch` mobile context proving
// the demo loads clean and its drag-to-orbit interaction works via real CDP touch, against the published
// surface. The web server is `shallot dev` (the standalone runtime, no editor), so the gate runs against
// the same path a user opens. This is full device testing: it needs a capable WebGPU GPU. In WSL the only
// adapter is software (llvmpipe), which fails shallot's device floor — `playwright.global-setup.ts` routes
// the run through `scripts/wsl-bridge.ts`'s host-GPU bridge there, so this reads `connectOptions` back from
// what it found (a worker process re-imports this file fresh, after global setup has already written it).
// Off WSL, and when the bridge's own prerequisites are absent, no endpoint file exists and this falls
// through to the local/native launch below — same as it always has, display-gated by the adapter-name skip
// in `test/touch-smoke.playwright.ts`.

const PORT = 3102;
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
        // standalone `shallot dev` over this project's manifest — `bunx` resolves the installed CLI. A cold
        // first vite build can run past 60s in CI; the warm cache serves in ~1s.
        command: `bunx shallot dev . --port ${PORT} --strict-port`,
        url: URL,
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
    },
    use: {
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
