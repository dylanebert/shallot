import { expect, type Locator, type Page, test } from "@playwright/test";
import { deriveDemosFromIframeSrcs } from "./demos";
import { classifyRendered } from "./rendered";

const sampleGrid = async (page: Page, canvas: Locator): Promise<number[]> => {
    const screenshot = await canvas.screenshot();
    return page.evaluate(async (base64) => {
        const binary = atob(base64);
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const surface = new OffscreenCanvas(64, 64);
        const context = surface.getContext("2d");
        if (!context) throw new Error("visualization gate: 2D screenshot context unavailable");
        context.drawImage(bitmap, 0, 0, 64, 64);
        bitmap.close();
        const rgba = context.getImageData(0, 0, 64, 64).data;
        const rgb: number[] = [];
        for (let at = 0; at < rgba.length; at += 4) rgb.push(rgba[at], rgba[at + 1], rgba[at + 2]);
        return rgb;
    }, screenshot.toString("base64"));
};

/** demos whose scene animates on its own (no input): the gate asserts the frame actually changes. */
const MOTION_DEMOS = new Set(["animation", "text"]);
const MOTION_WINDOW_MS = 1000;
// mean |Δ| per channel over the 64×64 grid: a moving cube or orbiting caption shifts a few percent of
// cells by tens of levels. Measured on nvidia/lovelace: text (slow orbit + a 0.24 m bob) reads ~0.3
// over 700 ms, the animation demo's five cubes several times that, and a parked scene ~0 — so the
// bar sits an order below the slowest shipped demo and well above a still frame's sampling noise.
const MOTION_THRESHOLD = 0.08;

function meanAbsDiff(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
}

test("visualization showcase — every demo renders a positive canvas", async ({ page }) => {
    // Derive the demo list from the built index.html iframes so a partially-added
    // demo can't ship unverified — a hand list drifts from the tree.
    await page.goto("/");
    const demos = deriveDemosFromIframeSrcs(
        await page
            .locator("iframe")
            .evaluateAll((iframes) => iframes.map((f) => (f as HTMLIFrameElement).src)),
    );
    // An empty derived population reads red, never vacuous-green — a missing index or a
    // broken parse must fail the gate, not pass it by skipping every demo.
    expect(demos.length, "index.html must list at least one demo iframe").toBeGreaterThan(0);

    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        const text = message.text();
        if (
            message.type() === "error" ||
            // the engine's degraded-boot warnings ride `console.warn`: a demo plugin dropped for a
            // missing dependency or a scene attribute nothing registered still renders a canvas, so
            // the pixel law alone reads green while the demo's headline behaviour is gone
            /draw .* skipped|GPUValidationError|WebGPU.*(?:error|failed)|Missing plugin dependency|is not registered|names no clip/i.test(
                text,
            )
        ) {
            errors.push(`[console.${message.type()}] ${text}`);
        }
    });

    for (const demo of demos) {
        await page.goto(`/demos/${demo}.html`);
        const canvas = page.locator("canvas");
        await expect(canvas).toBeVisible();
        await page.waitForFunction(() => {
            const canvas = document.querySelector("canvas");
            return canvas instanceof HTMLCanvasElement && canvas.width > 0 && canvas.height > 0;
        });
        let classification = classifyRendered(await sampleGrid(page, canvas));
        await expect
            .poll(
                async () => {
                    classification = classifyRendered(await sampleGrid(page, canvas));
                    return classification.rendered;
                },
                { message: `${demo}: canonical center-vs-corner rendered law`, timeout: 10_000 },
            )
            .toBe(true);
        expect(classification.rendered, `${demo}: ${JSON.stringify(classification)}`).toBe(true);
        expect(errors, errors.join("\n")).toEqual([]);
        if (MOTION_DEMOS.has(demo)) {
            // the demo's headline behaviour is motion, and a static canvas passes the rendered law —
            // the animation demo shipped with animators naming no clip, every cube parked, gate green.
            // Two grid samples across a beat of the loop must differ.
            const before = await sampleGrid(page, canvas);
            await page.waitForTimeout(MOTION_WINDOW_MS);
            const after = await sampleGrid(page, canvas);
            const diff = meanAbsDiff(before, after);
            expect(
                diff,
                `${demo}: canvas unchanged across ${MOTION_WINDOW_MS}ms (mean abs diff ${diff.toFixed(2)}, need > ${MOTION_THRESHOLD})`,
            ).toBeGreaterThan(MOTION_THRESHOLD);
        }
    }
});
