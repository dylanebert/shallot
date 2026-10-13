import { expect, test } from "playwright/test";

test("the HUD shows the turntable's live angle over the canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");

    const angle = page.locator('[data-hud="angle"]');
    await angle.waitFor();
    const first = await angle.textContent();
    await expect(angle).not.toHaveText(first ?? "");

    const canvas = await page.locator("canvas").boundingBox();
    const hud = await angle.boundingBox();
    expect(canvas && hud).toBeTruthy();
    expect(hud!.x).toBeGreaterThanOrEqual(canvas!.x);
    expect(hud!.y).toBeGreaterThanOrEqual(canvas!.y);
    expect(hud!.x + hud!.width).toBeLessThanOrEqual(canvas!.x + canvas!.width);
    expect(hud!.y + hud!.height).toBeLessThanOrEqual(canvas!.y + canvas!.height);
    expect(errors).toEqual([]);
});
