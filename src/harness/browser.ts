import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, sep } from "node:path";
import { CROSS_ORIGIN_ISOLATION } from "@dylanebert/shallot/vite";
import type { Page } from "playwright";
import floor from "./browser.json" with { type: "json" };
import { CAPTURE_CONTRACT } from "./capture";
import { servePage } from "./page";
import type { HarnessTarget, Verdict } from "./runtime";
import { MissingPremise, type VerdictMetadata } from "./verdict";

const PAGE_BUDGET_MS = 16_000;

function pageFailure(
    errors: readonly string[],
    cause?: unknown,
): (Error & { diagnostics?: NonNullable<VerdictMetadata["diagnostics"]> }) | null {
    if (errors.length === 0) return cause instanceof Error ? cause : null;
    const causeMessage = cause instanceof Error ? cause.message : undefined;
    const error = new Error(
        [
            ...(causeMessage === undefined ? [] : [causeMessage]),
            `page errors:\n${errors.join("\n")}`,
        ].join("\n"),
    ) as Error & { diagnostics: NonNullable<VerdictMetadata["diagnostics"]> };
    error.diagnostics = { pageErrors: [...errors] };
    return error;
}

function pageVerdict(value: unknown): value is Verdict & VerdictMetadata {
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

        const hosted = servePage(outDir, CROSS_ORIGIN_ISOLATION);
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
        const withPageDiagnostics = async <T>(work: Promise<T>): Promise<T> => {
            try {
                return await work;
            } catch (error) {
                throw pageFailure(errors, error) ?? new Error(String(error));
            }
        };
        await withPageDiagnostics(
            bounded(
                "page navigation",
                page.goto(`${hosted.origin}/${pagePath}`, {
                    waitUntil: "load",
                    timeout: remaining(),
                }),
            ),
        );
        await withPageDiagnostics(
            bounded(
                "page readiness",
                page.waitForFunction(
                    () =>
                        (window as Window & { __harness?: HarnessTarget }).__harness?.ready ===
                        true,
                    null,
                    { timeout: remaining() },
                ),
            ),
        );
        await withPageDiagnostics(
            bounded(
                "page driver",
                Promise.resolve().then(() => drive(page)),
            ),
        );

        const pageResult = await withPageDiagnostics(
            bounded(
                "the page verdict",
                page.evaluate(async () => {
                    const harness = (window as Window & { __harness?: HarnessTarget }).__harness;
                    if (typeof harness?.run !== "function") {
                        throw new Error("the page did not provide window.__harness.run()");
                    }
                    return harness.run();
                }),
            ),
        );
        const result = pageResult as Verdict & VerdictMetadata;
        if (!pageVerdict(result))
            throw new Error("the page returned no boolean window.__harness verdict");
        const failure = pageFailure(errors);
        return {
            ...result,
            ok: result.ok && failure === null,
            checks: [
                ...(result.checks ?? []),
                ...(failure === null
                    ? []
                    : [
                          {
                              name: "browser reported no page errors",
                              ok: false,
                              detail: failure.message,
                          },
                      ]),
            ],
            runtime: `chromium ${browser.version()} headless`,
            ...(failure === null
                ? {}
                : {
                      diagnostics: {
                          ...result.diagnostics,
                          ...failure.diagnostics,
                      },
                  }),
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
