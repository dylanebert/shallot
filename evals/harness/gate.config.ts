import { defineConfig } from "@playwright/test";
import { GLOBAL_TIMEOUT_MS, MAX_GATE_BUDGET_MS } from "./lib";

// Drives one task gate against the running project. Real Chrome (WebGPU needs a real GPU — the
// software adapter can't raster or run physics), a fixed viewport so fractional pixel coords are
// stable, layered timeouts so a blank canvas fails once the harness's own worst-case gate budget is
// exceeded rather than hanging — `timeout` and `globalTimeout` both derive from `harness/lib`'s one
// owner rather than a hand-picked number, so they move with it instead of drifting from it.
export default defineConfig({
    testDir: ".",
    testMatch: "gate.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: MAX_GATE_BUDGET_MS,
    globalTimeout: GLOBAL_TIMEOUT_MS,

    expect: { timeout: 5_000 },

    use: {
        trace: "off",
        video: "off",
        headless: false,
        viewport: { width: 1280, height: 800 },
        actionTimeout: 20_000,
        navigationTimeout: 30_000,
        launchOptions: {
            args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
    },

    projects: [{ name: "chromium", use: { channel: "chrome" } }],
});
