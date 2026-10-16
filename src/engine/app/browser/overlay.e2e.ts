import { expect, test } from "playwright/test";

test.use({ deviceScaleFactor: 1.333 });

test("overlay text keeps its CSS size on a 4K canvas at device scale 1.333", async ({ page }) => {
    await page.setViewportSize({ width: 2880, height: 1620 });
    await page.goto("/overlay.html");
    const text = page.locator("span");
    await expect(text).toHaveCSS("font-size", "14px");
    expect((await text.boundingBox())!.height).toBeCloseTo(20, 1);
    const overlay = (await text.locator("..").boundingBox())!;
    expect(overlay.width).toBeCloseTo(2880, 0);
    expect(overlay.height).toBeCloseTo(1620, 0);
});
