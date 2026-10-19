import { expect, test } from "playwright/test";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`instructions stay top left and toggles preserve the controls bounds at ${viewport.width}×${viewport.height}`, async ({
        browser,
    }) => {
        const context = await browser.newContext({ viewport, hasTouch: viewport.width === 390 });
        const page = await context.newPage();
        await page.goto("/");
        const picker = page.getByRole("button", { name: "Scene", exact: true });
        await picker.click();
        await page.getByRole("button", { name: "Contact", exact: true }).click();
        const help = page.getByText("WASD to roll", { exact: true });
        await expect(help).toBeVisible();
        expect.soft(await page.locator(".controls").getByText("WASD to roll").count()).toBe(0);
        const bounds = await help.boundingBox();
        expect.soft(bounds!.x).toBe(24);
        expect.soft(bounds!.y).toBe(24);
        await picker.click();
        await page.getByRole("button", { name: "Wheel", exact: true }).click();
        const panel = page.getByRole("region", { name: "Wheel controls", exact: true });
        await expect(panel).toBeVisible();
        const toggles = panel.getByRole("checkbox");
        for (let index = 1; index < (await toggles.count()); index++) {
            const previous = await toggles
                .nth(index - 1)
                .locator("..")
                .boundingBox();
            const current = await toggles.nth(index).locator("..").boundingBox();
            expect.soft(current!.y - previous!.y - previous!.height).toBe(4);
        }
        const before = await panel.boundingBox();
        expect(before!.x + before!.width).toBe(viewport.width - 24);
        expect(before!.y).toBe(24);
        for (const name of ["Steering", "Steering Limit", "Steering Limit", "Steering"]) {
            await panel.getByRole("checkbox", { name, exact: true }).click();
            expect.soft(await panel.boundingBox()).toEqual(before);
        }
        await context.close();
    });

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
