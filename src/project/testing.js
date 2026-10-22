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

/**
 * Holds a page's animation frames from the first request after `ready` holds. A held frame runs only on
 * `run(count)`, which runs the held callbacks for `count` browser frames, 1 by default, and resolves after the
 * last; under `runApp`, each is one `world.step` and its draw. Input events, locator actions and screenshots
 * otherwise wait on scene draws, which a software adapter spaces by seconds. Call before navigating.
 * `ready` is a CSS selector or a self-contained in-page predicate, checked at each request until it first
 * holds once the standard loading screen is gone. A reload or navigation starts a new document, whose frames run freely until `ready` holds again.
 * @param {import("playwright/test").Page} page
 * @param {string | (() => boolean)} ready
 * @returns {Promise<{ run(count?: number): Promise<void> }>}
 */
export async function holdFrames(page, ready) {
    await page.addInitScript(
        ({ selector, predicate }) => {
            const holds = selector
                ? () => document.querySelector(selector) !== null
                : new Function(`return (${predicate})()`);
            // runApp mounts its UI under the loading screen, which a held frame would never remove.
            const isReady = () =>
                document.querySelector("[data-shallot-loading]") === null && holds();
            const request = window.requestAnimationFrame.bind(window);
            const cancel = window.cancelAnimationFrame.bind(window);
            const held = new Map();
            let holding = false;
            let next = -1;
            window.requestAnimationFrame = (callback) => {
                holding ||= Boolean(isReady());
                if (!holding) return request(callback);
                held.set(next, callback);
                return next--;
            };
            window.cancelAnimationFrame = (id) => (id < 0 ? held.delete(id) : cancel(id));
            Object.defineProperty(window, "__shallotRunHeldFrames", {
                value: async (/** @type {number} */ count) => {
                    for (let frame = 0; frame < count; frame++)
                        await new Promise((resolve) =>
                            request((time) => {
                                // A callback may cancel another due this frame; requests it makes wait for the next.
                                for (const id of [...held.keys()]) {
                                    const callback = held.get(id);
                                    if (!callback) continue;
                                    held.delete(id);
                                    callback(time);
                                }
                                resolve(undefined);
                            }),
                        );
                },
            });
        },
        typeof ready === "string"
            ? { selector: ready, predicate: "" }
            : { selector: "", predicate: ready.toString() },
    );
    return {
        run: (count = 1) =>
            page.evaluate(
                (count) =>
                    /** @type {{ __shallotRunHeldFrames(count: number): Promise<void> }} */ (
                        /** @type {unknown} */ (window)
                    ).__shallotRunHeldFrames(count),
                count,
            ),
    };
}

/** Playwright fixtures: `test.use(pipelineWarnings)` fails on late pipelines, including manually created pages. */
export const pipelineWarnings = {
    /**
     * @param {{ browser: import("playwright/test").Browser }} fixtures
     * @param {(browser: import("playwright/test").Browser) => Promise<void>} use
     */
    browser: async ({ browser }, use) => {
        /** @type {string[]} */
        const warnings = [];
        /** @type {Set<import("playwright/test").BrowserContext>} */
        const contexts = new Set();
        /** @param {import("playwright/test").ConsoleMessage} message */
        const onConsole = (message) => {
            if (message.type() === "warning" && message.text().startsWith("[shallot] Pipeline ") &&
                message.text().includes("created after the precompile drain.")) warnings.push(message.text());
        };
        /** @param {import("playwright/test").BrowserContext} context */
        const watch = (context) => {
            if (!contexts.has(context)) {
                contexts.add(context);
                context.on("console", onConsole);
                context.once("close", () => contexts.delete(context));
            }
            return context;
        };
        /** @type {import("playwright/test").Browser["newContext"]} */
        const newContext = async (options) => watch(await browser.newContext(options));
        /** @type {import("playwright/test").Browser["newPage"]} */
        const newPage = async (options) => {
            const page = await browser.newPage(options);
            watch(page.context());
            return page;
        };
        const guarded = new Proxy(browser, {
            get(target, key) {
                if (key === "newContext") return newContext;
                if (key === "newPage") return newPage;
                const value = Reflect.get(target, key, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        });
        try {
            await use(guarded);
        } finally {
            for (const context of contexts) context.off("console", onConsole);
            if (warnings.length) throw new Error(warnings.join("\n"));
        }
    },
    /**
     * @param {{ page: import("playwright/test").Page }} fixtures
     * @param {(page: import("playwright/test").Page) => Promise<void>} use
     */
    page: async ({ page }, use) => {
        /** @type {string[]} */
        const warnings = [];
        /** @param {import("playwright/test").ConsoleMessage} message */
        const onConsole = (message) => {
            if (message.type() === "warning" && message.text().startsWith("[shallot] Pipeline ") &&
                message.text().includes("created after the precompile drain.")) warnings.push(message.text());
        };
        const context = page.context();
        context.on("console", onConsole);
        try {
            await use(page);
        } finally {
            context.off("console", onConsole);
            if (warnings.length) throw new Error(warnings.join("\n"));
        }
    },
};
