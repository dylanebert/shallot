import { expect, test } from "playwright/test";

test("a Box Stack cube follows the cursor grab and release removes the mouse and joint", async ({
    page,
}) => {
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    expect(await page.evaluate(() => (window as any).marker())).toBe(0);
    const result = await page.evaluate(() => (window as any).drag());
    expect(result.markerDuring).toBe(true);
    expect(result.markerNoShadow).toBe(true);
    result.markerPosition.forEach((value: number, axis: number) => {
        expect(value).toBeCloseTo(result.target[axis], 5);
    });
    if (process.env.SHALLOT_CAPTURE_DIR) {
        await page.evaluate(() => (window as any).captureGrab());
        await page.screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/grab.png` });
    }
    const released = await page.evaluate(() => (window as any).finishDrag());
    expect(released.marker).toBe(0);
    expect(result.picked).toBe(result.body);
    expect(result.projected[0] - result.pixel[0]).toBeGreaterThan(60);
    expect(result.projected[0] - result.pixel[0]).toBeLessThan(110);
    expect(Math.abs(result.projected[1] - result.pixel[1])).toBeLessThan(15);
    for (const pose of result.cameraDuring) expect(pose).toEqual(result.cameraBefore);
    expect(result.cameraAfter).toEqual(result.cameraBefore);
    expect(released.mouse).toBe(0);
    expect(released.joint).toBe(0);
});

test("hover adds an outline and leaving removes it", async ({ page }) => {
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    expect(await page.evaluate(() => (window as any).hover())).toEqual({
        added: true,
        removed: true,
    });
});

test("a left drag over empty space orbits without grabbing a body", async ({ page }) => {
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    const result = await page.evaluate(() => (window as any).emptyDrag());
    expect(result.picked).toBe(0);
    expect(result.mouse).toBe(0);
    expect(result.after).not.toEqual(result.before);
    expect(result.bodyAfter[0]).toBe(result.bodyBefore[0]);
    expect(result.bodyAfter[2]).toBe(result.bodyBefore[2]);
});
