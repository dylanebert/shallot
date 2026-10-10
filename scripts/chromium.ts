import type { PlaywrightTestConfig } from "playwright/test";
import { CEILING } from "./test-tiers";

export const CHROMIUM_USE = {
    channel: "chromium" as const,
    launchOptions: {
        args: [
            "--enable-unsafe-webgpu",
            "--enable-features=WebGPUDeveloperFeatures",
            "--enable-webgpu-developer-features",
            "--enable-gpu",
            ...(process.platform === "linux" ? ["--use-webgpu-adapter=swiftshader"] : []),
        ],
    },
};

export const BROWSER_CONFIG = {
    testMatch: "**/*.e2e.ts",
    globalTimeout: CEILING.browser,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        ...CHROMIUM_USE,
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
} satisfies PlaywrightTestConfig;

export const WEB_SERVER_CONFIG = {
    reuseExistingServer: false,
    timeout: CEILING.startup,
};
