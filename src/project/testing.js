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

/**
 * Install WebGPU globals and open a check's app. The engine warns once on a non-real adapter;
 * the check owns that notice, so captured warnings must equal exactly the adapter's notice.
 * @param {Parameters<typeof import("@dylanebert/shallot").createApp>[0]} config
 * @returns {ReturnType<typeof import("@dylanebert/shallot").createApp>}
 */
export async function openGpuApp(config) {
    const { setupGlobals } = await import("@dylanebert/shallot/webgpu");
    const { createApp } = await import("@dylanebert/shallot");
    const { adapterNotice } = await import("../engine/runtime/gpu");
    await setupGlobals();
    const notices = /** @type {string[]} */ ([]);
    const warn = console.warn;
    console.warn = (...args) => notices.push(args.map(String).join(" "));
    let app;
    try {
        app = await createApp(config);
    } finally {
        console.warn = warn;
    }
    const notice = adapterNotice(app.world.gpuIfAvailable?.adapter);
    const expected = notice === undefined ? [] : [notice];
    if (JSON.stringify(notices) !== JSON.stringify(expected)) {
        app.dispose();
        throw new Error(
            `Expected adapter notices ${JSON.stringify(expected)}, observed ${JSON.stringify(notices)}`,
        );
    }
    return app;
}

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
