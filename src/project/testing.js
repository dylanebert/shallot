export const CEILING = /** @type {const} */ ({
    cheap: 2500,
    gpu: 10_000,
    node: 120_000,
    browser: 300_000,
    startup: 120_000,
});

/** Local iteration budgets in milliseconds. */
export const BUDGET = /** @type {const} */ ({
    cheap: 250,
    gpu: 1000,
    node: 20_000,
    browser: 60_000,
});

/** CI tolerates host variance up to the hang ceiling; local runs enforce speed budgets. */
export const TIMEOUT = process.env.CI ? CEILING : { ...BUDGET, startup: CEILING.startup };

// Linux stays on SwiftShader even where Mesa's llvmpipe is installed: Chromium only accepts a non-SwiftShader
// Dawn adapter that imports external images, which on desktop Linux needs opaque-FD semaphores llvmpipe lacks.
const chromiumArgs =
    process.platform === "linux"
        ? [
              "--use-angle=swiftshader",
              "--no-sandbox",
              "--enable-unsafe-webgpu",
              "--enable-features=Vulkan",
              "--use-vulkan=swiftshader",
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

/** @satisfies {import("playwright/test").PlaywrightTestConfig} */
export const BROWSER_CONFIG = {
    testMatch: "**/*.e2e.ts",
    retries: process.env.CI ? 1 : 0,
    use: {
        trace: "on-first-retry",
        browserName: "chromium",
        channel: "chromium",
        launchOptions,
    },
};

export const WEB_SERVER_CONFIG = {
    reuseExistingServer: false,
    timeout: CEILING.startup,
};
