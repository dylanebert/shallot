import { defineConfig } from "@playwright/test";

export default defineConfig({
    testDir: "./flows",
    testMatch: "*.pw.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 120000,

    use: {
        trace: "off",
        video: "off",
        headless: false,
        viewport: { width: 1920, height: 1080 },
    },

    projects: [
        {
            name: "chromium",
            use: { channel: "chrome" },
        },
    ],
});
