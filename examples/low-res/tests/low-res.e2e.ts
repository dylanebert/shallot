import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);
test("a panel change reaches the stepped frame", async ({ page }) => {
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    const before = await page.evaluate(() => (window as any).captureScene());
    await page.getByRole("slider", { name: "Colour levels" }).fill("4");
    const after = await page.evaluate(() => (window as any).captureScene());
    expect(after.rgba).not.toEqual(before.rgba);
    let invalid = 0;
    for (let i = 0; i < after.rgba.length; i++)
        if (i % 4 !== 3 && ![0, 85, 170, 255].includes(after.rgba[i])) invalid++;
    expect(invalid).toBe(0);
    expect(before.height).toBe(360);
    await page.getByRole("slider", { name: "Resolution height" }).fill("240");
    const resized = await page.evaluate(() => (window as any).captureScene());
    expect(resized.height).toBe(240);
});
test("390 by 844 touch page keeps controls inside the canvas", async ({ browser }) => {
    const context = await browser.newContext({
        viewport: { width: 390, height: 844 },
        hasTouch: true,
    });
    const page = await context.newPage();
    try {
        await page.goto("http://127.0.0.1:4265/tests/stepped.html");
        await page.waitForFunction(() => (window as any).ready);
        const panel = page.getByRole("region", { name: "Low-res controls" });
        const bounds = (await panel.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(390);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(844);
        const before = await page.evaluate(() => (window as any).captureScene());
        await page.getByRole("checkbox", { name: "Dither" }).tap();
        await expect(page.getByRole("checkbox", { name: "Dither" })).not.toBeChecked();
        const after = await page.evaluate(() => (window as any).captureScene());
        expect(after.rgba).not.toEqual(before.rgba);
    } finally {
        await context.close();
    }
});
