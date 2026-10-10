import type { PlaywrightTestConfig } from "playwright/test";
import { CEILING } from "./test-tiers";

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
                "--use-angle=d3d11-warp",
                "--use-webgpu-adapter=d3d11",
            ]
          : [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-webgpu-developer-features",
                "--enable-gpu",
            ];

export const CHROMIUM_USE = {
    channel: "chromium" as const,
    launchOptions: { args: chromiumArgs },
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
