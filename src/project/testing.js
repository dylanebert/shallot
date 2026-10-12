export const CEILING = /** @type {const} */ ({
    cheap: 250,
    gpu: 1000,
    node: 20_000,
    browser: 60_000,
    startup: 120_000,
});

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
    use: {
        browserName: "chromium",
        channel: "chromium",
        launchOptions,
    },
};

export const WEB_SERVER_CONFIG = {
    reuseExistingServer: false,
    timeout: CEILING.startup,
};
