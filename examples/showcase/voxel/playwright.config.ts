import { defineConfig } from "@playwright/test";

// The voxel showcase's own browser driver — bring-your-own, as a real user would. shallot exports no
// Playwright harness (bun ships `bun test` and tells you to bring Playwright); the reusable part is the
// gate logic in `src/gate.ts`, written against the published surface. The web server is `shallot dev` (the
// standalone runtime, no editor), so the gate runs against the same path a user opens. This is full device
// testing: it needs a capable WebGPU GPU. In WSL the only adapter is software (llvmpipe), which fails
// shallot's device floor, so the gate is display-gated there — the same posture `bun run capture` takes.

const PORT = 3100;
const URL = `http://localhost:${PORT}`;

export default defineConfig({
    testDir: "./test",
    fullyParallel: false,
    workers: 1,
    reporter: [["list"]],
    timeout: 120_000,
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
    },
});
