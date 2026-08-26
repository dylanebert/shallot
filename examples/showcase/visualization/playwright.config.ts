import { existsSync, readFileSync } from "node:fs";
import { defineConfig } from "@playwright/test";
import { ENDPOINT_FILE } from "./playwright.global-setup";

// This project's own browser driver — bring-your-own, as a real user would. In WSL the only adapter is
// software (llvmpipe), which fails shallot's device floor — `playwright.global-setup.ts` routes the run
// through `scripts/wsl-bridge.ts`'s host-GPU bridge there, so this reads `connectOptions` back from what it
// found (a worker process re-imports this file fresh, after global setup has already written it). Off WSL,
// and when the bridge's own prerequisites are absent, no endpoint file exists and this falls through to the
// local/native launch below — same as it always has. `test/touch-smoke.playwright.ts` display-gates
// itself with an adapter-name skip; `test/visualization.playwright.ts` has none of its own and simply
// inherits whichever adapter this config resolves.

const PORT = 3118;
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
        command: `bun run build && bunx vite preview --port ${PORT} --strictPort`,
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
