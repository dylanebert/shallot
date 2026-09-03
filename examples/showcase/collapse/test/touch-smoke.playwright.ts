import { isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { expect, test } from "@playwright/test";
import { adapterName, SOFTWARE } from "./gpu-adapter";
import { oneFingerDrag } from "./touch-drag";

// S5's showcase touch smoke (`shallot-mobile-controls` spec): the collapse demo loads clean and its
// primary interaction (drag-to-orbit) works under a real `hasTouch` mobile context, driven through CDP
// `Input.dispatchTouchEvent` — the same integration-honest instrument gym's own touch gate uses, never
// Playwright's synthetic `dispatchEvent` (bypasses the touch-action/listener path this asserts). Runs by
// path — `cd examples/showcase/collapse && bunx playwright test test/touch-smoke.playwright.ts` —
// display-gated (WSL bridge, `playwright.global-setup.ts`), never part of `bun run test`.

test.use({
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 },
});

// how much a 0-255 channel sample must move, averaged over a coarse grid, to count as "the frame
// changed" — comfortably above screenshot encoding noise, well under what an actual ~60px-of-travel orbit
// drag produces against the collapse structure.
const FrameChangeThreshold = 3;

async function sampleCanvas(page: import("@playwright/test").Page): Promise<number[]> {
    const canvas = page.locator("canvas").first();
    const screenshot = await canvas.screenshot();
    return page.evaluate(async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const surface = new OffscreenCanvas(32, 32);
        const ctx = surface.getContext("2d");
        if (!ctx) throw new Error("touch smoke: 2D context unavailable");
        ctx.drawImage(bitmap, 0, 0, 32, 32);
        bitmap.close();
        return Array.from(ctx.getImageData(0, 0, 32, 32).data);
    }, screenshot.toString("base64"));
}

function meanAbsDiff(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
}

test("collapse showcase — loads clean and orbits by touch", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        if (m.type() === "error" || isDegradedBootMessage(m.text())) {
            errors.push(`[console.${m.type()}] ${m.text()}`);
        }
    });

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`collapse touch smoke adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await page.waitForFunction(() => {
        const c = document.querySelector("canvas");
        return c instanceof HTMLCanvasElement && c.width > 0 && c.height > 0;
    });

    const before = await sampleCanvas(page);

    const cdp = await page.context().newCDPSession(page);
    const box = await canvas.boundingBox();
    if (!box) throw new Error("touch smoke: canvas has no bounding box");
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    await oneFingerDrag(cdp, { x: cx - 80, y: cy }, { x: cx + 80, y: cy + 40 });

    // poll instead of a fixed sleep — the orbit's smoothed pose (`smoothLerp`, extras/orbit) settles
    // over a few frames, not instantly, so wait on the condition itself: the sampled frame diverging.
    await expect
        .poll(async () => meanAbsDiff(before, await sampleCanvas(page)), {
            message: "a one-finger drag should visibly rotate the orbit camera",
        })
        .toBeGreaterThan(FrameChangeThreshold);

    expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
});
