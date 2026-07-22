import { defineConfig } from "@playwright/test";

// Drives one task gate against the running project. Real Chrome (WebGPU needs a real GPU — the
// software adapter can't raster or run physics), a fixed viewport so fractional pixel coords are
// stable, layered timeouts so a blank canvas fails in a minute rather than hanging.
export default defineConfig({
    testDir: ".",
    testMatch: "gate.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 90_000,
    globalTimeout: 180_000,

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
