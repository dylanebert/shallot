import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, sep } from "node:path";
import type { Page } from "playwright";
import { type AdapterFacts, classifyAdapter } from "../engine/runtime";
import floor from "./browser.json" with { type: "json" };
import { CAPTURE_CONTRACT } from "./capture";
import { servePage } from "./page";
import type { HarnessTarget, Verdict } from "./runtime";
import { MissingPremise, type VerdictMetadata } from "./verdict";

const PAGE_BUDGET_MS = 16_000;

/** Refuse a browser adapter that cannot support a real-device claim. */
export function browserAdapterRefusal(facts: AdapterFacts): string | null {
    const adapter = classifyAdapter(facts);
    return adapter.class === "real"
        ? null
        : `browser seat unavailable: ${adapter.reason ?? adapter.class}`;
}

async function browserAdapterFacts(page: Page): Promise<AdapterFacts> {
    return page.evaluate(async () => {
        const gpu = navigator.gpu;
        if (!gpu) return { present: false };
        const adapter = await gpu.requestAdapter();
        if (!adapter) return { present: false };
        const info = adapter.info as (GPUAdapterInfo & { isFallbackAdapter?: boolean }) | undefined;
        return {
            present: true,
            info: {
                vendor: info?.vendor,
                architecture: info?.architecture,
                device: info?.device,
                description: info?.description,
                isFallbackAdapter: info?.isFallbackAdapter,
            },
        };
    });
}

function pageVerdict(value: unknown): value is Verdict {
    return (
        value !== null && typeof value === "object" && typeof (value as Verdict).ok === "boolean"
    );
}

/**
 * Build a page entry with Shallot's TGSL transform, serve it as an isolated localhost document, then let
 * the caller drive a real Playwright page before returning its `window.__harness.run()` verdict.
 *
 * @example
 * ```
 * const verdict = await runBrowserCheck(pageEntry, async (page) => {
 *     await page.keyboard.press("Space");
 * });
 * ```
 */
export async function runBrowserCheck(
    entry: string,
    drive: (page: Page) => void | Promise<void>,
): Promise<Verdict & VerdictMetadata> {
    const deadline = performance.now() + PAGE_BUDGET_MS;
    const teardownDeadline = deadline + 3_000;
    const remaining = () => Math.max(1, deadline - performance.now());
    const bounded = <T>(label: string, work: Promise<T>): Promise<T> => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const timeout = new Promise<never>((_, reject) => {
            timer = setTimeout(
                () => reject(new Error(`${label} exceeded the browser check deadline`)),
                remaining(),
            );
        });
        return Promise.race([work, timeout]).finally(() => clearTimeout(timer));
    };

    const pageEntry = realpathSync(entry);
    const root = dirname(pageEntry);
    const pagePath = relative(root, pageEntry).split(sep).join("/");
    const outDir = mkdtempSync(join(tmpdir(), "shallot-browser-"));
    let server: ReturnType<typeof Bun.serve> | undefined;
    let browser: import("playwright").Browser | undefined;
    try {
        const [{ build }, { default: typegpu }] = await Promise.all([
            import("vite"),
            import("unplugin-typegpu/vite"),
        ]);
        await bounded(
            "the TGSL page build",
            build({
                root,
                configFile: false,
                logLevel: "error",
                plugins: [typegpu() as unknown as import("vite").Plugin],
                build: {
                    target: "esnext",
                    outDir,
                    emptyOutDir: true,
                    rollupOptions: { input: pageEntry },
                },
            }),
        );

        const hosted = servePage(outDir);
        server = hosted.server;
        const { chromium } = await import("playwright");
        try {
            browser = await bounded(
                "Chromium launch",
                chromium.launch({
                    headless: true,
                    channel: floor.channel as "chromium",
                    args: [...floor.args, "--enable-gpu"],
                    timeout: remaining(),
                }),
            );
        } catch (error) {
            throw new MissingPremise(
                `browser seat unavailable: Chromium did not launch${error instanceof Error ? `: ${error.message}` : `: ${String(error)}`}`,
            );
        }
        const page = await bounded(
            "newPage",
            browser.newPage({
                viewport: {
                    width: CAPTURE_CONTRACT.width,
                    height: CAPTURE_CONTRACT.height,
                },
                deviceScaleFactor: CAPTURE_CONTRACT.deviceScale,
            }),
        );
        page.setDefaultTimeout(remaining());
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await bounded(
            "page navigation",
            page.goto(`${hosted.origin}/${pagePath}`, { waitUntil: "load", timeout: remaining() }),
        );
        const facts = await bounded("browser adapter observation", browserAdapterFacts(page));
        const refusal = browserAdapterRefusal(facts);
        if (refusal !== null) throw new MissingPremise(refusal);

        await bounded(
            "page readiness",
            page.waitForFunction(
                () => (window as Window & { __harness?: HarnessTarget }).__harness?.ready === true,
                null,
                { timeout: remaining() },
            ),
        );
        await bounded(
            "page driver",
            Promise.resolve().then(() => drive(page)),
        );
        if (errors.length > 0) throw new Error(`the browser page threw:\n${errors.join("\n")}`);

        const result = await bounded(
            "the page verdict",
            page.evaluate(async () => {
                const harness = (window as Window & { __harness?: HarnessTarget }).__harness;
                if (typeof harness?.run !== "function") {
                    throw new Error("the page did not provide window.__harness.run()");
                }
                return harness.run();
            }),
        );
        if (!pageVerdict(result))
            throw new Error("the page returned no boolean window.__harness verdict");
        if (errors.length > 0) throw new Error(`the browser page threw:\n${errors.join("\n")}`);
        return {
            ...result,
            runtime: `chromium ${browser.version()} headless`,
            hardware: classifyAdapter(facts).identity,
        };
    } finally {
        if (browser) {
            let timer: ReturnType<typeof setTimeout> | undefined;
            await Promise.race([
                browser.close(),
                new Promise<void>((resolve) => {
                    timer = setTimeout(resolve, Math.max(0, teardownDeadline - performance.now()));
                }),
            ]).catch(() => {});
            if (timer !== undefined) clearTimeout(timer);
        }
        server?.stop(true);
        rmSync(outDir, { recursive: true, force: true });
    }
}
