import { expect, type Page, test } from "@playwright/test";
import { deriveDemosFromIframeSrcs } from "./demos";
import { classifyRendered } from "./rendered";

// S5's showcase touch smoke (`shallot-mobile-controls` spec): every visualization gallery demo loads
// clean, and its first demo's drag-to-orbit interaction works, under a real `hasTouch` mobile context —
// driven through CDP `Input.dispatchTouchEvent`, the same integration-honest instrument gym's own touch
// gate uses, never Playwright's synthetic `dispatchEvent` (bypasses the touch-action/listener path this
// asserts). One representative demo carries the interaction assertion (every demo shares the same
// `OrbitPlugin` boot, `src/boot.ts`); the rest are covered by the clean-load loop, matching
// `visualization.playwright.ts`'s own "every demo renders a positive canvas" shape. Runs by path —
// `cd examples/showcase/visualization && bunx playwright test test/touch-smoke.playwright.ts` —
// display-gated by `playwright.global-setup.ts` routing through the WSL bridge for real GPU access, plus
// this file's own adapter-name skip below for a seat where neither the bridge nor a native real GPU is
// available (mirrors `roads/test/touch-smoke.playwright.ts`); never part of `bun run test`.

test.use({
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 },
});

// Software rasterizers by the name they report in `GPUAdapterInfo` — the same display-gate pattern
// `roads/test/touch-smoke.playwright.ts` uses (that file's header has the full rationale).
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

const sampleGrid = async (page: Page): Promise<number[]> => {
    const canvas = page.locator("canvas");
    const screenshot = await canvas.screenshot();
    return page.evaluate(async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const surface = new OffscreenCanvas(64, 64);
        const context = surface.getContext("2d");
        if (!context) throw new Error("touch smoke: 2D screenshot context unavailable");
        context.drawImage(bitmap, 0, 0, 64, 64);
        bitmap.close();
        const rgba = context.getImageData(0, 0, 64, 64).data;
        const rgb: number[] = [];
        for (let at = 0; at < rgba.length; at += 4) rgb.push(rgba[at], rgba[at + 1], rgba[at + 2]);
        return rgb;
    }, screenshot.toString("base64"));
};

function meanAbsDiff(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
}

test("visualization showcase — every demo loads clean, one orbits by touch", async ({ page }) => {
    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`visualization touch smoke adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    const demos = deriveDemosFromIframeSrcs(
        await page
            .locator("iframe")
            .evaluateAll((iframes) => iframes.map((f) => (f as HTMLIFrameElement).src)),
    );
    expect(demos.length, "index.html must list at least one demo iframe").toBeGreaterThan(0);

    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(`[console.error] ${message.text()}`);
    });

    for (const demo of demos) {
        await page.goto(`/demos/${demo}.html`);
        const canvas = page.locator("canvas");
        await expect(canvas).toBeVisible();
        await page.waitForFunction(() => {
            const c = document.querySelector("canvas");
            return c instanceof HTMLCanvasElement && c.width > 0 && c.height > 0;
        });
        let classification = classifyRendered(await sampleGrid(page));
        if (!classification.rendered) {
            await expect
                .poll(
                    async () => {
                        classification = classifyRendered(await sampleGrid(page));
                        return classification.rendered;
                    },
                    {
                        message: `${demo}: canonical center-vs-corner rendered law`,
                        timeout: 10_000,
                    },
                )
                .toBe(true);
        }
        expect(classification.rendered, `${demo}: ${JSON.stringify(classification)}`).toBe(true);
    }

    // one representative demo carries the interaction assertion — the first in the derived list.
    const [first] = demos;
    await page.goto(`/demos/${first}.html`);
    const canvas = page.locator("canvas");
    await expect(canvas).toBeVisible();
    const before = await sampleGrid(page);

    const cdp = await page.context().newCDPSession(page);
    const box = await canvas.boundingBox();
    if (!box) throw new Error("touch smoke: canvas has no bounding box");
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const id = 1;
    await cdp.send("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [{ x: cx - 80, y: cy, id }],
    });
    await cdp.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ x: cx + 80, y: cy + 40, id }],
    });
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });

    // poll instead of a fixed sleep — the orbit's smoothed pose (`smoothLerp`, extras/orbit) settles over
    // a few frames, not instantly, so wait on the condition itself: the sampled frame actually diverging.
    await expect
        .poll(async () => meanAbsDiff(before, await sampleGrid(page)), {
            message: `${first}: a one-finger drag should visibly rotate the orbit camera`,
        })
        .toBeGreaterThan(3);

    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});
