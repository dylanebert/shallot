import { expect, test } from "playwright/test";
import { categories } from "../src/scenes";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`instructions stay top left and toggles preserve the controls width and top-left at ${viewport.width}×${viewport.height}`, async ({
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
        for (let index = 0; index < (await toggles.count()); index++) {
            const group = toggles.nth(index).locator("../..");
            const dependents = group.locator(":scope > .dependents");
            if (await dependents.locator(":scope > .group").count()) {
                const toggle = await toggles.nth(index).locator("..").boundingBox();
                const child = await dependents.locator(":scope > .group").first().boundingBox();
                expect.soft(child!.x - toggle!.x).toBe(8);
                expect.soft(child!.y).toBeGreaterThanOrEqual(toggle!.y + toggle!.height + 4);
            }
        }
        const before = await panel.boundingBox();
        expect(before!.height).toBe(128);
        expect(before!.width).toBeLessThanOrEqual(300);
        expect(before!.x + before!.width).toBe(viewport.width - 24);
        expect(before!.y).toBe(24);
        let height = before!.height;
        for (const name of ["Steering", "Steering Limit", "Steering Limit", "Steering"]) {
            const toggle = panel.getByRole("checkbox", { name, exact: true });
            await toggle.click();
            const after = (await panel.boundingBox())!;
            expect
                .soft({ x: after.x, y: after.y, width: after.width })
                .toEqual({ x: before!.x, y: before!.y, width: before!.width });
            if (await toggle.isChecked()) expect.soft(after.height).toBeGreaterThan(height);
            else expect.soft(after.height).toBeLessThan(height);
            height = after.height;
        }
        await context.close();
    });

    test(`picker arrows stay fixed across every scene name at ${viewport.width}×${viewport.height}`, async ({
        browser,
    }) => {
        const context = await browser.newContext({ viewport });
        const page = await context.newPage();
        await page.goto("/");
        const arrows = [
            page.getByRole("button", { name: "Previous scene" }),
            page.getByRole("button", { name: "Next scene" }),
        ];
        const before = await Promise.all(arrows.map((arrow) => arrow.boundingBox()));
        for (const sample of categories.flatMap((category) => category.scenes)) {
            await page.getByRole("button", { name: "Scene", exact: true }).click();
            await page.getByRole("button", { name: sample.name, exact: true }).click();
            expect
                .soft(await Promise.all(arrows.map((arrow) => arrow.boundingBox())), sample.name)
                .toEqual(before);
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
