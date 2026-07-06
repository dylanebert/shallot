import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
    testDir: ".",
    testMatch: "page.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 600_000,

    use: {
        trace: "off",
        video: "off",
        headless: false,
    },

    projects: [
        {
            name: "chromium",
            use: {
                ...devices["Desktop Chrome"],
                channel: "chrome",
                launchOptions: {
                    // Bypass Chrome's WebGPU timestamp-query quantization for tight
                    // microbench measurements. Harmless for the gym path.
                    args: [
                        "--enable-features=WebGPUDeveloperFeatures",
                        "--enable-unsafe-webgpu",
                        "--enable-dawn-features=allow_unsafe_apis",
                    ],
                },
            },
        },
    ],
});
