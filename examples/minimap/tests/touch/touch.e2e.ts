import { mkdirSync } from "node:fs";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);
test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });
test("390×844: the square corner stays inside the canvas and above working touch controls", async ({
    page,
    context,
}) => {
    await page.goto("http://127.0.0.1:4197/tests/touch.html");
    await page.waitForFunction(() => "sample" in window);
    await page.evaluate(() => (window as any).reset());
    const sample = () => page.evaluate(() => (window as any).sample(1));
    const before = await sample();
    const session = await context.newCDPSession(page);
    await session.send("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [
            { id: 1, x: 90, y: 620 },
            { id: 2, x: 280, y: 450 },
        ],
    });
    await sample();
    await session.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [
            { id: 1, x: 90, y: 550 },
            { id: 2, x: 330, y: 450 },
        ],
    });
    for (let i = 0; i < 20; i++) await sample();
    const after = await sample();
    expect(
        Math.hypot(after.position[0] - before.position[0], after.position[2] - before.position[2]),
    ).toBeGreaterThan(0.1);
    expect(after.yaw).not.toBe(before.yaw);
    const r = await page.evaluate(() => (window as any).corner());
    const canvas = (await page.locator("canvas").boundingBox())!;
    expect(r.width).toBe(r.height);
    expect(r.width).toBeGreaterThan(100);
    expect(r.x).toBeGreaterThanOrEqual(0);
    expect(r.y).toBeGreaterThanOrEqual(0);
    expect(r.x + r.width).toBeLessThanOrEqual(canvas.width);
    expect(r.y + r.height).toBeLessThanOrEqual(canvas.height);
    for (const control of await page.locator("button, [data-touch-stick]").all()) {
        const b = await control.boundingBox();
        if (!b) continue;
        expect(
            r.x + r.width <= b.x ||
                b.x + b.width <= r.x ||
                r.y + r.height <= b.y ||
                b.y + b.height <= r.y,
        ).toBe(true);
    }
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
        await page.evaluate(() => (window as any).reset());
        await session.send("Input.dispatchTouchEvent", {
            type: "touchStart",
            touchPoints: [{ id: 1, x: 90, y: 620 }],
        });
        await sample();
        mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/phone.png` });
    }
    await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
});
