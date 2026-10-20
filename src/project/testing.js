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

/**
 * CI hang ceilings for one browser test and one of its waits. A software adapter spaced frames by 21 s on
 * Windows, and the slowest modelled test took 24 s. A hang and its retry leave the rest of the slowest subject
 * (85 s of a 135 s Windows run) inside {@link CEILING}.browser, so a red reports its line on both attempts.
 */
export const BROWSER_CEILING = /** @type {const} */ ({
    test: 90_000,
    expect: 60_000,
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
    // Input, locators and the frame loop all wait on browser frames, which a software adapter spaces by seconds.
    ...(process.env.CI && {
        timeout: BROWSER_CEILING.test,
        expect: { timeout: BROWSER_CEILING.expect },
    }),
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
