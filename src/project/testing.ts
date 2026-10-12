import type { PlaywrightTestConfig } from "playwright/test";

export const CEILING = {
    cheap: 250,
    gpu: 1000,
    node: 20_000,
    browser: 60_000,
    startup: 120_000,
} as const;

const chromiumArgs =
    process.platform === "linux"
        ? [
              "--use-angle=swiftshader",
              "--no-sandbox",
              "--enable-unsafe-webgpu",
              "--enable-features=Vulkan",
          ]
        : process.platform === "win32"
          ? [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-webgpu-developer-features",
                "--enable-gpu",
                "--ignore-gpu-blocklist",
            ]
          : [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-webgpu-developer-features",
                "--enable-gpu",
            ];

const launchOptions =
    process.platform === "win32"
        ? { args: chromiumArgs, ignoreDefaultArgs: ["--enable-unsafe-swiftshader"] }
        : { args: chromiumArgs };

export const BROWSER_CONFIG = {
    testMatch: "**/*.e2e.ts",
    globalTimeout: CEILING.browser,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        channel: "chromium",
        launchOptions,
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
} satisfies PlaywrightTestConfig;

export const WEB_SERVER_CONFIG = {
    reuseExistingServer: false,
    timeout: CEILING.startup,
};
