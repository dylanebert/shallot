import { test } from "@playwright/test";
import type { Page, Locator } from "@playwright/test";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import type { Annotation, Step, Manifest, StepOpts, FlowDef } from "./core";
import { S } from "./selectors";

export interface FlowContext {
    page: Page;
    step: (id: string, opts?: StepOpts) => Promise<Step>;
    assert: (description: string, condition: boolean) => void;
    act: {
        click: (target: string | Locator) => Promise<void>;
        fill: (target: string | Locator, value: string) => Promise<void>;
        key: (key: string) => Promise<void>;
        wait: (ms: number) => Promise<void>;
        waitFor: (selector: string, opts?: { timeout?: number }) => Promise<void>;
    };
}

const PORT = process.env.CAPTURE_PORT || "3004";
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

async function waitForEditor(page: Page): Promise<void> {
    await page.waitForSelector(S.editor, { timeout: 30000 });
    await page.waitForSelector(S.outliner, { timeout: 10000 });
    await page.waitForSelector(S.inspector, { timeout: 10000 });
    await page.waitForSelector(S.canvas, { timeout: 10000 });

    for (let i = 0; i < 120; i++) {
        const ready = await page.evaluate(() => {
            const canvas = document.querySelector("canvas");
            return canvas != null && canvas.width > 0 && canvas.height > 0;
        });
        if (ready) {
            await page.waitForTimeout(500);
            return;
        }
        await page.waitForTimeout(250);
    }
    throw new Error("Editor canvas did not initialize within 30s");
}

function toLocator(page: Page, target: string | Locator): Locator {
    return typeof target === "string" ? page.locator(target) : target;
}

export function defineFlow(def: FlowDef, fn: (ctx: FlowContext) => Promise<void>): void {
    test(def.name, async ({ page }) => {
        const base = process.env.CAPTURE_OUT;
        if (!base) throw new Error("CAPTURE_OUT env var required");

        const outDir = join(base, def.name);
        mkdirSync(outDir, { recursive: true });

        await page.goto(`http://localhost:${PORT}`);
        await waitForEditor(page);

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
                    timeout: opts?.timeout ?? 10000,
                });
            },
        };

        const ctx: FlowContext = { page, step, assert, act };
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
