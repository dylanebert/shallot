import type { Page } from "@playwright/test";
import { PNG } from "pngjs";

// The shared driver every task gate builds on. A gate is a Playwright test that boots the agent's
// running project, observes it through the canvas the engine draws into (pixels + synthetic input —
// no cooperation from the agent's code required, so a gate grades any project the same way), asserts
// the task's positive claim, and emits one envelope on stdout. The grader decodes that envelope; the
// test's own pass/fail signals only whether the gate could run, never whether the task was met.
//
// Self-contained: no imports outside @playwright/test + pngjs, so the WSL→Windows staging copies this
// file with the gate and nothing else.

export interface Assertion {
    name: string;
    ok: boolean;
    detail?: string;
}

export interface EvalEnvelope {
    // the task's positive claim held: booted, rendered, and every assertion true
    ok: boolean;
    // the project served a page with a sized canvas
    booted: boolean;
    // the canvas drew a non-blank frame within the settle window
    rendered: boolean;
    assertions: Assertion[];
    errors?: string[];
}

export interface Rgb {
    r: number;
    g: number;
    b: number;
}

const MARK = "__EVAL_RESULT__";

export function url(): string {
    const u = process.env.EVAL_URL;
    if (!u) throw new Error("EVAL_URL not set");
    return u;
}

export function emit(e: EvalEnvelope): void {
    console.log(`${MARK}${JSON.stringify(e)}${MARK}`);
}

// screenshot the first canvas and decode it to RGBA pixels
export async function shot(page: Page): Promise<PNG> {
    const buf = await page.locator("canvas").first().screenshot();
    return PNG.sync.read(buf);
}

// average color over a fractional rectangle (0..1 in each axis) of the frame
export function region(png: PNG, fx0: number, fy0: number, fx1: number, fy1: number): Rgb {
    const x0 = Math.floor(fx0 * png.width);
    const x1 = Math.max(x0 + 1, Math.floor(fx1 * png.width));
    const y0 = Math.floor(fy0 * png.height);
    const y1 = Math.max(y0 + 1, Math.floor(fy1 * png.height));
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            const i = (y * png.width + x) * 4;
            r += png.data[i];
            g += png.data[i + 1];
            b += png.data[i + 2];
            n++;
        }
    }
    return { r: r / n, g: g / n, b: b / n };
}

// mean absolute RGB difference between two frames over a sampled grid, 0..255 — the magnitude of
// visible change (a view rotation, a color swap) between two moments
export function diff(a: PNG, b: PNG): number {
    const w = Math.min(a.width, b.width);
    const h = Math.min(a.height, b.height);
    const step = Math.max(1, Math.floor(Math.min(w, h) / 64));
    let sum = 0;
    let n = 0;
    for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
            const ia = (y * a.width + x) * 4;
            const ib = (y * b.width + x) * 4;
            sum +=
                Math.abs(a.data[ia] - b.data[ib]) +
                Math.abs(a.data[ia + 1] - b.data[ib + 1]) +
                Math.abs(a.data[ia + 2] - b.data[ib + 2]);
            n += 3;
        }
    }
    return sum / n;
}

// vertical centroid (0 top .. 1 bottom) of pixels matching a predicate — where a distinctly-colored
// object sits on screen, for tracking motion between frames
export function centroidY(png: PNG, match: (c: Rgb) => boolean): number | null {
    let sum = 0;
    let n = 0;
    for (let y = 0; y < png.height; y++) {
        for (let x = 0; x < png.width; x++) {
            const i = (y * png.width + x) * 4;
            if (match({ r: png.data[i], g: png.data[i + 1], b: png.data[i + 2] })) {
                sum += y;
                n++;
            }
        }
    }
    return n > 0 ? sum / png.height / n : null;
}

// is the frame non-blank — does it carry visible structure rather than a single flat clear color
export function structured(png: PNG): boolean {
    const c = region(png, 0.4, 0.4, 0.6, 0.6);
    const corner = region(png, 0, 0, 0.15, 0.15);
    const spread = Math.abs(c.r - corner.r) + Math.abs(c.g - corner.g) + Math.abs(c.b - corner.b);
    return spread > 12;
}

// navigate to the project and wait for the engine to draw a non-blank, settled frame. A cold vite
// server can re-optimize deps mid-load and strand the first navigation, so retry the goto once (the
// flows launcher's proven shape). The first structured frame is NOT the settled one: measured on the
// baseline projects (2026-07-13), exactly one visible transition (lighting/shadow completing, delta
// ~9–12) lands within ~500ms of it, after which every 500ms delta is exactly 0.00 — the settled
// render is bit-static. So after structure appears, keep sampling until two consecutive shots diff
// below 0.5 (epsilon over the measured 0.00 floor); a gate's first read is then the settled scene,
// never the pre-settle transient.
export async function boot(
    page: Page,
    settleMs = 20_000,
): Promise<{ booted: boolean; rendered: boolean }> {
    const open = async () => {
        await page.goto(url(), { timeout: 30_000 });
        await page.waitForSelector("canvas", { timeout: 20_000 });
    };
    try {
        await open();
    } catch {
        await page.waitForTimeout(1000);
        await open();
    }
    const size = await page
        .locator("canvas")
        .first()
        .evaluate((c) => (c as HTMLCanvasElement).width * (c as HTMLCanvasElement).height)
        .catch(() => 0);
    if (!size) return { booted: false, rendered: false };

    const deadline = Date.now() + settleMs;
    let prev: PNG | null = null;
    while (Date.now() < deadline) {
        const cur = await shot(page);
        if (structured(cur)) {
            if (prev && diff(prev, cur) < 0.5) return { booted: true, rendered: true };
            prev = cur;
        }
        await page.waitForTimeout(500);
    }
    // structured but never stable: an animated scene. Rendered — the gate's own assertions decide
    // whether the motion is correct (a spinning scaffold fails its "static" assertion, not boot).
    return { booted: true, rendered: prev != null };
}
