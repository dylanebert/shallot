import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Algorithm controls and vertex readouts fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const select = async (name: string) => {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Collision", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
            await page.evaluate(() =>
                (window as unknown as { controlsKey(): number }).controlsKey(),
            );
        };
        await select("Shape Distance");
        const panel = page.locator(".controls");
        const bounds = await panel.boundingBox();
        const radius = panel.getByLabel("radius A, m", { exact: true });
        await expect(radius).toBeDisabled();
        await panel.getByLabel("shape A", { exact: true }).selectOption("0");
        await expect(radius).toBeEnabled();
        expect(await panel.boundingBox()).toEqual(bounds);
        await pointerControl(page, panel.getByLabel("show indices", { exact: true }));
        const index = panel.getByRole("slider", { name: "index", exact: true });
        await expect(index).toBeDisabled();
        await pointerControl(page, panel.getByLabel("draw simplex", { exact: true }));
        await expect(index).toBeEnabled();
        await bodyHasFocus(page);
        await expect(
            panel.getByRole("region", { name: "Distance and vertex identity", exact: true }),
        ).toBeVisible();
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        await select("Time of Impact");
        expect(
            await panel.getByLabel("shape A", { exact: true }).locator("option").allTextContents(),
        ).toEqual(["box", "capsule", "triangle"]);
        await expect(panel.getByLabel("State", { exact: true })).toHaveText("hit");
        await select("Distance Debug");
        await expect(
            panel.getByRole("slider", { name: "simplex index", exact: true }),
        ).toBeVisible();
        await expect(panel.getByLabel("Simplex 0 A : B", { exact: true })).not.toBeEmpty();
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        await select("Shape Cast Debug");
        await expect(panel).toHaveCount(0);
    });
