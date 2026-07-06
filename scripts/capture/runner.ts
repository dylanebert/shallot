import type { Locator, Page } from "@playwright/test";
import { test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { PNG } from "pngjs";
import type {
    Annotation,
    FlowDef,
    Manifest,
    SampleOpts,
    SampleResult,
    Step,
    StepOpts,
} from "./core";
import { S } from "./selectors";

export interface FlowContext {
    page: Page;
    step: (id: string, opts?: StepOpts) => Promise<Step>;
    assert: (description: string, condition: boolean) => void;
    /** read the viewport canvas back as pixels — the render-band gate (non-blank, selection outline) */
    sample: (opts?: SampleOpts) => Promise<SampleResult>;
    /** navigate to an editor server on `port` and wait for it to mount + size its canvas — the
     *  per-specimen step of a `target: "manual"` sweep across several editor servers */
    openEditor: (port: number) => Promise<void>;
    act: {
        click: (target: string | Locator) => Promise<void>;
        fill: (target: string | Locator, value: string) => Promise<void>;
        key: (key: string) => Promise<void>;
        wait: (ms: number) => Promise<void>;
        waitFor: (selector: string, opts?: { timeout?: number }) => Promise<void>;
    };
}

const PORT = process.env.CAPTURE_PORT || "3004";
const APP_PORT = process.env.CAPTURE_APP_PORT || "3005";
const SETTLE_MS = 300;

async function resolveBounds(
    page: Page,
    selector: string,
): Promise<{ x: number; y: number; width: number; height: number } | null> {
    const el = await page.$(selector);
    if (!el) return null;
    const box = await el.boundingBox();
    if (!box) return null;
    return {
        x: Math.round(box.x),
        y: Math.round(box.y),
        width: Math.round(box.width),
        height: Math.round(box.height),
    };
}

// Editor chrome mounts together (one Svelte app), so the panels resolve near-instantly once `.editor`
// is up; the real wait is the engine sizing its canvas after the first frame. Each ceiling fits the
// per-test budget (capture.pw.config.ts: 15s) and throws a specific error before that generic timeout.
async function waitForEditor(page: Page): Promise<void> {
    await page.waitForSelector(S.editor, { timeout: 6000 });
    await page.waitForSelector(S.outliner, { timeout: 3000 });
    await page.waitForSelector(S.inspector, { timeout: 3000 });
    await page.waitForSelector(S.canvas, { timeout: 3000 });
    await page.waitForFunction(
        () => {
            const canvas = document.querySelector("canvas");
            return canvas != null && canvas.width > 0 && canvas.height > 0;
        },
        null,
        { timeout: 6000 },
    );
    // the sized canvas proves the chrome mounted, not that the engine built — the control seam answers
    // { error: "No engine running" } until build() resolves, and a flow's first read races that on a
    // slow machine. Poll the seam itself so every flow starts against a live State.
    await waitForEngine(page);
    await page.waitForTimeout(300);
}

async function waitForEngine(page: Page, timeout = 15_000): Promise<void> {
    const origin = new URL(page.url()).origin;
    const deadline = Date.now() + timeout;
    for (;;) {
        const body = await page.request
            .get(`${origin}/__api/entities`)
            .then((r) => r.json())
            .catch(() => null);
        if (Array.isArray(body)) return;
        if (Date.now() > deadline) throw new Error(`engine did not build within ${timeout}ms`);
        await page.waitForTimeout(250);
    }
}

// a cold vite server can re-optimize deps mid-load, which strands the first navigation (504 Outdated
// Optimize Dep on modules, or vite's own reload aborting the goto) — a fresh goto after the re-bundle
// loads clean, so retry once before failing the flow.
async function openAndWait(page: Page, url: string): Promise<void> {
    try {
        await page.goto(url);
        await waitForEditor(page);
    } catch {
        await page.waitForTimeout(1000);
        await page.goto(url);
        await waitForEditor(page);
    }
}

// the standalone-app counterpart to waitForEditor: no editor chrome, just the run() app's canvas sized
// after its first frame. The fixture exposes a `__survive` read seam once `run()` resolves.
async function waitForApp(page: Page): Promise<void> {
    await page.waitForSelector(S.canvas, { timeout: 6000 });
    await page.waitForFunction(
        () => {
            const canvas = document.querySelector("canvas");
            return canvas != null && canvas.width > 0 && canvas.height > 0;
        },
        null,
        { timeout: 6000 },
    );
    await page.waitForTimeout(300);
}

function toLocator(page: Page, target: string | Locator): Locator {
    return typeof target === "string" ? page.locator(target) : target;
}

export function defineFlow(def: FlowDef, fn: (ctx: FlowContext) => Promise<void>): void {
    test(def.name, async ({ page }) => {
        if (def.timeout) test.setTimeout(def.timeout);
        // the editor routes its runtime firehose — script throws, GPU validation, build failures — to the
        // browser console (editor-ui.md), invisible to a flow's pixel asserts. Forward error/warning console
        // output + uncaught errors to the test stdout so a capture failure carries its cause, not just a
        // blank verdict. (Editor-vocabulary errors that surface only in the DOM — diagnose Issues, banners —
        // are read by the flow that gates on them; this catches everything that reaches the console.)
        page.on("console", (m) => {
            const level = m.type();
            if (level === "error" || level === "warning")
                console.log(`[browser:${level}] ${m.text()}`);
        });
        page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
        const base = process.env.CAPTURE_OUT;
        if (!base) throw new Error("CAPTURE_OUT env var required");

        const outDir = join(base, def.name);
        mkdirSync(outDir, { recursive: true });

        if (def.target === "app") {
            await page.goto(`http://localhost:${APP_PORT}`);
            await waitForApp(page);
        } else if (def.target !== "manual") {
            // ?save=off: capture runs the editor in ephemeral mode so flows never write the fixture scenes
            await openAndWait(page, `http://localhost:${PORT}?save=off`);
        }

        const steps: Step[] = [];
        const t0 = Date.now();

        const step = async (id: string, opts?: StepOpts): Promise<Step> => {
            const filename = `${id}.png`;
            const annotations: Annotation[] = [];

            const highlights = opts?.highlight
                ? Array.isArray(opts.highlight)
                    ? opts.highlight
                    : [opts.highlight]
                : [];
            for (const sel of highlights) {
                const bounds = await resolveBounds(page, sel);
                if (bounds) annotations.push({ type: "region", selector: sel, bounds });
            }

            if (opts?.click) {
                const bounds = await resolveBounds(page, opts.click);
                if (bounds) annotations.push({ type: "click", selector: opts.click, bounds });
            }

            if (opts?.labels) {
                for (const [sel, label] of Object.entries(opts.labels)) {
                    const bounds = await resolveBounds(page, sel);
                    if (bounds) annotations.push({ type: "label", selector: sel, bounds, label });
                }
            }

            const screenshotOpts: Parameters<Page["screenshot"]>[0] = {
                path: join(outDir, filename),
                type: "png",
            };
            if (opts?.clip) {
                const bounds = await resolveBounds(page, opts.clip);
                if (bounds) screenshotOpts.clip = bounds;
            }
            await page.screenshot(screenshotOpts);

            const vp = page.viewportSize()!;
            const captured: Step = {
                id,
                screenshot: filename,
                timestamp: Date.now() - t0,
                viewport: { width: vp.width, height: vp.height },
                annotations,
            };
            steps.push(captured);
            return captured;
        };

        const assert = (description: string, condition: boolean): void => {
            if (!condition) throw new Error(`Assertion failed: ${description}`);
        };

        // A WebGPU canvas doesn't preserve its drawing buffer, so an in-page drawImage/getImageData reads
        // back black. page.screenshot is the compositor capture — the only path that returns real pixels
        // (the same path the step screenshots use). Decode the canvas-region PNG and sample it.
        const sample = async (opts?: SampleOpts): Promise<SampleResult> => {
            const region = opts?.region ?? { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
            const bgTol = opts?.bgTol ?? 24;
            const accent = opts?.accent ?? null;
            const accentTol = opts?.accentTol ?? 40;

            const el = await page.$(S.canvas);
            const box = el && (await el.boundingBox());
            if (!box || box.width < 1 || box.height < 1) {
                return { nonBackground: 0, accent: 0, background: [0, 0, 0], total: 0 };
            }
            const {
                width: w,
                height: h,
                data,
            } = PNG.sync.read(await page.screenshot({ clip: box }));
            const rgb = (x: number, y: number): [number, number, number] => {
                const i = (y * w + x) * 4;
                return [data[i], data[i + 1], data[i + 2]];
            };
            const bg = rgb(0, 0);
            const rx = Math.floor(region.x * w);
            const ry = Math.floor(region.y * h);
            const rw = Math.max(1, Math.floor(region.w * w));
            const rh = Math.max(1, Math.floor(region.h * h));
            let total = 0;
            let nonBg = 0;
            let acc = 0;
            for (let y = ry; y < ry + rh && y < h; y++) {
                for (let x = rx; x < rx + rw && x < w; x++) {
                    const [r, g, b] = rgb(x, y);
                    total++;
                    if (
                        Math.abs(r - bg[0]) > bgTol ||
                        Math.abs(g - bg[1]) > bgTol ||
                        Math.abs(b - bg[2]) > bgTol
                    ) {
                        nonBg++;
                    }
                    if (
                        accent &&
                        Math.abs(r - accent[0]) <= accentTol &&
                        Math.abs(g - accent[1]) <= accentTol &&
                        Math.abs(b - accent[2]) <= accentTol
                    ) {
                        acc++;
                    }
                }
            }
            return {
                nonBackground: total ? nonBg / total : 0,
                accent: total ? acc / total : 0,
                background: bg,
                total,
            };
        };

        const act = {
            click: async (target: string | Locator) => {
                await toLocator(page, target).click();
                await page.waitForTimeout(SETTLE_MS);
            },
            fill: async (target: string | Locator, value: string) => {
                const loc = toLocator(page, target);
                await loc.fill(value);
                await page.waitForTimeout(SETTLE_MS);
            },
            key: async (key: string) => {
                await page.keyboard.press(key);
                await page.waitForTimeout(SETTLE_MS);
            },
            wait: async (ms: number) => {
                await page.waitForTimeout(ms);
            },
            waitFor: async (selector: string, opts?: { timeout?: number }) => {
                await page.waitForSelector(selector, {
                    timeout: opts?.timeout ?? 5000,
                });
            },
        };

        const openEditor = async (port: number): Promise<void> => {
            await openAndWait(page, `http://localhost:${port}?save=off`);
        };

        const ctx: FlowContext = { page, step, assert, sample, act, openEditor };
        await fn(ctx);

        const manifest: Manifest = {
            flow: def.name,
            scene: def.scene,
            timestamp: new Date().toISOString(),
            viewport: page.viewportSize()!,
            steps,
        };
        writeFileSync(join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    });
}
