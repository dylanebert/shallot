import { test as base, expect, type Page } from "playwright/test";

const noticePrefix = "__shallot_browser_startup_failure__: ";
type FailureKind = "page error" | "startup alert";
type ExpectedFailure = { kind: FailureKind; pattern: RegExp; matched: boolean };
const expectedFailures = new WeakMap<Page, ExpectedFailure[]>();

export type { Page };
export { expect };

function expectFailure(page: Page, kind: FailureKind, pattern: RegExp): void {
    const failures = expectedFailures.get(page) ?? [];
    failures.push({ kind, pattern, matched: false });
    expectedFailures.set(page, failures);
}

export function expectPageError(page: Page, pattern: RegExp): void {
    expectFailure(page, "page error", pattern);
}

export function expectStartupAlert(page: Page, pattern: RegExp): void {
    expectFailure(page, "startup alert", pattern);
}

/** Run `body`, then `release`; a release that fails after the body failed never replaces the body's error. */
export async function releasing<T>(
    body: () => Promise<T>,
    release: () => Promise<unknown>,
): Promise<T> {
    let result: T;
    try {
        result = await body();
    } catch (error) {
        await release().catch(() => {});
        throw error;
    }
    await release();
    return result;
}

export const test = base.extend({
    page: async ({ page }, use) => {
        const failures = expectedFailures.get(page) ?? [];
        expectedFailures.set(page, failures);
        const unexpected: string[] = [];
        let closing: Promise<void> | undefined;
        const fail = (message: string): void => {
            unexpected.push(message);
            closing ??= page.close().catch(() => {});
        };
        const consumeExpected = (kind: FailureKind, message: string): boolean => {
            const expected = failures.find((entry) => {
                entry.pattern.lastIndex = 0;
                return entry.kind === kind && entry.pattern.test(message);
            });
            if (!expected) return false;
            expected.matched = true;
            return true;
        };
        const onPageError = (error: Error): void => {
            const message = `${error.name}: ${error.message}`;
            if (!consumeExpected("page error", error.message)) fail(`page error: ${message}`);
        };
        const onConsole = (message: { type(): string; text(): string }): void => {
            if (message.type() !== "error" || !message.text().startsWith(noticePrefix)) return;
            const notice = message.text().slice(noticePrefix.length);
            if (!consumeExpected("startup alert", notice)) fail(`startup alert: ${notice}`);
        };

        // A crashed page or a lost browser closes the page under the test, whose own error then reads only
        // "Target page, context or browser has been closed"; record which one did it.
        const onCrash = (): void => {
            unexpected.push("page crash");
        };
        const onDisconnected = (): void => {
            unexpected.push("browser disconnected");
        };
        const browser = page.context().browser();
        page.on("pageerror", onPageError);
        page.on("console", onConsole);
        page.on("crash", onCrash);
        browser?.on("disconnected", onDisconnected);
        await page.addInitScript((prefix) => {
            const reported = new WeakSet<HTMLElement>();
            const inspect = (): void => {
                for (const notice of document.querySelectorAll<HTMLElement>(
                    '[role="alert"], [data-shallot-loading]',
                )) {
                    const text = notice.textContent?.replace(/\s+/g, " ").trim() ?? "";
                    // A software adapter's notice is accepted, as browser observations accept software adapters.
                    if (!text || /^\w+ adapter: /.test(text) || reported.has(notice)) continue;
                    reported.add(notice);
                    console.error(`${prefix}${text}`);
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
            page.off("crash", onCrash);
            browser?.off("disconnected", onDisconnected);
            expectedFailures.delete(page);
        }
        await closing;
        const missing = failures
            .filter((failure) => !failure.matched)
            .map((failure) => `expected ${failure.kind} matching ${failure.pattern}`);
        if (unexpected.length || missing.length)
            throw new Error([...unexpected, ...missing].join("\n"));
        if (testFailure) throw testFailure;
    },
});
