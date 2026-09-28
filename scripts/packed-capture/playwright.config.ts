import type { PlaywrightTestConfig } from "playwright/test";
import { CHROMIUM_ARGS } from "../chromium";

export const config = {
    testDir: ".",
    testMatch: "**/*.e2e.ts",
    timeout: 120_000,
    globalTimeout: 6_000,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        launchOptions: { args: CHROMIUM_ARGS },
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
} satisfies PlaywrightTestConfig;

export default config;
