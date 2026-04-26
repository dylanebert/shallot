import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
    testDir: ".",
    testMatch: "gpu.pw.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 120000,

    webServer: {
        command: "bun dev --port 3002 --strictPort",
        cwd: "../../../../examples/gym",
        port: 3002,
        reuseExistingServer: true,
        timeout: 30000,
    },

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
            },
        },
    ],
});
