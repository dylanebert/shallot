import { expect, test } from "playwright/test";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`scene list stays within ${viewport.width}×${viewport.height} and dismisses`, async ({
        browser,
    }) => {
        const context = await browser.newContext({ viewport, hasTouch: viewport.width === 390 });
        const page = await context.newPage();
        await page.goto("/");
        const trigger = page.getByRole("button", { name: "Scene", exact: true });
        await trigger.click();
        const list = page.locator("#scene-list");
        await expect(list).toBeVisible();
        const bounds = await list.boundingBox();
        expect(bounds).not.toBeNull();
        expect(bounds!.x).toBeGreaterThanOrEqual(24);
        expect(bounds!.y).toBeGreaterThanOrEqual(24);
        expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width - 24);
        expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height - 24);
        await page.mouse.click(4, 4);
        await expect(list).toBeHidden();
        await trigger.click();
        await page.keyboard.press("Escape");
        await expect(list).toBeHidden();
        await expect(trigger).toBeFocused();
        await context.close();
    });
}
