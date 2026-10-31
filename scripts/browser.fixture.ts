import { test as base, expect } from "playwright/test";

const noticePrefix = "__shallot_browser_startup_failure__: ";

export type { Page } from "playwright/test";
export { expect };

export const test = base.extend({
    page: async ({ page }, use) => {
        let failure: Error | undefined;
        let closing: Promise<void> | undefined;
        const fail = (error: Error): void => {
            if (failure) return;
            failure = new Error(`Browser reported an error: ${error.message}`, { cause: error });
            closing = page.close().catch(() => {});
        };
        const onPageError = (error: Error): void => {
            fail(new Error(`page error: ${error.name}: ${error.message}`, { cause: error }));
        };
        const onConsole = (message: { type(): string; text(): string }): void => {
            if (message.text().startsWith("Shallot browser adapter: ")) console.log(message.text());
            if (message.type() !== "error" || !message.text().startsWith(noticePrefix)) return;
            fail(new Error(`Shallot startup notice: ${message.text().slice(noticePrefix.length)}`));
        };

        page.on("pageerror", onPageError);
        page.on("console", onConsole);
        await page.addInitScript((prefix) => {
            if (navigator.gpu) {
                const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
                navigator.gpu.requestAdapter = async (options) => {
                    const adapter = await requestAdapter(options);
                    console.log(
                        `Shallot browser adapter: ${JSON.stringify(
                            adapter
                                ? {
                                      vendor: adapter.info.vendor,
                                      architecture: adapter.info.architecture,
                                      device: adapter.info.device,
                                      description: adapter.info.description,
                                      isFallbackAdapter: adapter.info.isFallbackAdapter,
                                  }
                                : null,
                        )}`,
                    );
                    return adapter;
                };
            }
            const inspect = (): void => {
                for (const notice of document.querySelectorAll<HTMLElement>(
                    '[role="alert"], [data-shallot-loading]',
                )) {
                    const text = notice.textContent?.replace(/\s+/g, " ").trim() ?? "";
                    if (/^(?:Something went wrong|Unsupported configuration)/.test(text)) {
                        console.error(`${prefix}${text}`);
                        return;
                    }
                }
            };
            new MutationObserver(inspect).observe(document, {
                childList: true,
                subtree: true,
                characterData: true,
            });
            inspect();
        }, noticePrefix);

        let testFailure: unknown;
        try {
            await use(page);
        } catch (error) {
            testFailure = error;
        } finally {
            page.off("pageerror", onPageError);
            page.off("console", onConsole);
        }
        await closing;
        if (failure) throw failure;
        if (testFailure) throw testFailure;
    },
});
