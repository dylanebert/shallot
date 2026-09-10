import { describe, expect, test } from "bun:test";
import {
    HARNESS_PROBE_SCRIPT,
    harnessInstallMs,
    type ResourceEntry,
    type ResourceTiming,
    summarizeResourceTiming,
    TIMINGS_INIT_SCRIPT,
} from "./verify";

// The `--timings` probes' real-browser red-proofs, split out of `verify.test.ts` as its own by-path gate
// (browser probes stay out of the default suite for speed): each launches Chromium, which the default
// seconds-scale suite must not do —
// and did flake once inside it. Run when you touch the probes:
//
//     bun test ./packages/shallot/bin/verify.probes.ts
//
// The sentinel left behind in `verify.test.ts` is the pure half — `installHarnessProbe`,
// `summarizeResourceTiming`, the saturation rendering — which pins the mechanism without a browser.

// A probe that lies is worse than none, so each is driven through a genuine Playwright page.
describe("real-browser red-proofs (b) + (c)", () => {
    let hasChromium = false;
    try {
        require.resolve("playwright");
        hasChromium = true;
    } catch {
        hasChromium = false;
    }

    test.skipIf(!hasChromium)(
        "(b) a delayed __harness install reports ≈N ms with the probe installed, and nothing without it",
        async () => {
            const { chromium } = await import("playwright");
            const DelayMs = 300;
            const server = Bun.serve({
                port: 0,
                fetch() {
                    return new Response(
                        `<!doctype html><script>setTimeout(() => { window.__harness = { ready: true }; }, ${DelayMs});</script>`,
                        { headers: { "content-type": "text/html" } },
                    );
                },
            });
            const browser = await chromium.launch({ headless: true });
            try {
                // red: the probe installed before navigation — the delayed install reports ≈N ms, not
                // the ±500ms-quantized poll `stepWait` would give.
                const withProbe = await browser.newPage();
                await withProbe.addInitScript({ content: HARNESS_PROBE_SCRIPT });
                const navStart = Date.now();
                await withProbe.goto(`http://localhost:${server.port}/`);
                await withProbe.waitForFunction(
                    () => typeof window.__harness !== "undefined",
                    null,
                    { timeout: 5_000 },
                );
                const installedAt = (await withProbe.evaluate(
                    () => (window as unknown as { __harnessInstallAt?: number }).__harnessInstallAt,
                )) as number | undefined;
                const measured = harnessInstallMs(navStart, installedAt ?? null);
                expect(measured).not.toBeNull();
                expect(measured as number).toBeGreaterThanOrEqual(DelayMs - 100);
                expect(measured as number).toBeLessThan(DelayMs + 2_000);
                await withProbe.close();

                // the point of the red-proof: the identical page, minus the probe, gives no install
                // signal at all — a silently wrong (missing) reading, not a caught error.
                const withoutProbe = await browser.newPage();
                await withoutProbe.goto(`http://localhost:${server.port}/`);
                await withoutProbe.waitForFunction(
                    () => typeof window.__harness !== "undefined",
                    null,
                    { timeout: 5_000 },
                );
                const noInstalledAt = (await withoutProbe.evaluate(
                    () => (window as unknown as { __harnessInstallAt?: number }).__harnessInstallAt,
                )) as number | undefined;
                expect(noInstalledAt).toBeUndefined();
                expect(harnessInstallMs(navStart, noInstalledAt ?? null)).toBeNull();
                await withoutProbe.close();
            } finally {
                await browser.close();
                server.stop(true);
            }
        },
        20_000,
    );

    test.skipIf(!hasChromium)(
        "(c) one extra module import moves the resource count by exactly 1",
        async () => {
            const { chromium } = await import("playwright");
            const server = Bun.serve({
                port: 0,
                fetch(req) {
                    const path = new URL(req.url).pathname;
                    const js = { headers: { "content-type": "application/javascript" } };
                    const html = { headers: { "content-type": "text/html" } };
                    if (path === "/a.js") return new Response("export const a = 1;", js);
                    if (path === "/b.js") return new Response("export const b = 2;", js);
                    if (path === "/base.html")
                        return new Response(`<script type="module" src="/a.js"></script>`, html);
                    if (path === "/variant.html")
                        return new Response(
                            `<script type="module" src="/a.js"></script>` +
                                `<script type="module" src="/b.js"></script>`,
                            html,
                        );
                    return new Response("not found", { status: 404 });
                },
            });
            const browser = await chromium.launch({ headless: true });
            try {
                const countFor = async (path: string): Promise<number> => {
                    const page = await browser.newPage();
                    await page.goto(`http://localhost:${server.port}${path}`, {
                        waitUntil: "networkidle",
                    });
                    const entries = (await page.evaluate(() =>
                        performance
                            .getEntriesByType("resource")
                            .map((e) => ({ name: e.name, duration: e.duration })),
                    )) as ResourceEntry[];
                    await page.close();
                    return summarizeResourceTiming(entries).count;
                };
                const base = await countFor("/base.html");
                const variant = await countFor("/variant.html");
                expect(variant - base).toBe(1);
            } finally {
                await browser.close();
                server.stop(true);
            }
        },
        20_000,
    );

    // the count-moves-by-1 proof above runs at 2 resources, far under the browser's 250-entry default
    // buffer — it is satisfiable without the property that matters, since the real gym page fetches
    // hundreds and the count this readout must report is precisely the one that saturates. Both real
    // measured sides read exactly 250 before TIMINGS_INIT_SCRIPT raised the buffer.
    test.skipIf(!hasChromium)(
        "(c) past the 250-entry default buffer, the count saturates without the raise and is real with it",
        async () => {
            const { chromium } = await import("playwright");
            const Modules = 400;
            const server = Bun.serve({
                port: 0,
                fetch(req) {
                    const path = new URL(req.url).pathname;
                    if (path.endsWith(".js"))
                        return new Response("export const x = 1;", {
                            headers: { "content-type": "application/javascript" },
                        });
                    const tags = Array.from(
                        { length: Modules },
                        (_, i) => `<script type="module" src="/m${i}.js"></script>`,
                    ).join("");
                    return new Response(tags, { headers: { "content-type": "text/html" } });
                },
            });
            const browser = await chromium.launch({ headless: true });
            const read = async (raise: boolean): Promise<ResourceTiming> => {
                const page = await browser.newPage();
                if (raise) await page.addInitScript({ content: TIMINGS_INIT_SCRIPT });
                await page.goto(`http://localhost:${server.port}/`, { waitUntil: "networkidle" });
                const entries = (await page.evaluate(() =>
                    performance
                        .getEntriesByType("resource")
                        .map((e) => ({ name: e.name, duration: e.duration })),
                )) as ResourceEntry[];
                await page.close();
                return summarizeResourceTiming(entries);
            };
            try {
                // red: the default buffer flattens 400 real requests to the spec's 250 and says nothing.
                const capped = await read(false);
                expect(capped.count).toBe(250);
                expect(
                    summarizeResourceTiming(Array(250).fill({ name: "", duration: 0 }), 10, 250),
                ).toHaveProperty("saturated", true);

                // green: the raise reports the real count, and the readout says it isn't a floor.
                const raised = await read(true);
                expect(raised.count).toBeGreaterThan(250);
                expect(raised.saturated).toBe(false);
            } finally {
                await browser.close();
                server.stop(true);
            }
        },
        30_000,
    );
});
