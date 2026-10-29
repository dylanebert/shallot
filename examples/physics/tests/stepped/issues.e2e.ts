import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Issues scenes, controls and regression readouts fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const select = async (name: string) => {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Issues", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
        };
        await select("Crash");
        const panel = page.locator(".controls");
        await pointerControl(page, panel.getByRole("button", { name: "Add Joint", exact: true }));
        await bodyHasFocus(page);
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        for (const name of [
            "Capsule Mesh",
            "Convex Jitter",
            "Heightfield",
            "Huge Box",
            "Hull Crash",
            "Multiple Prismatic",
            "GMod Wheel Stack",
            "s&box mover",
            "Slide Twist Off Center Shape",
        ]) {
            await select(name);
            await expect(panel).toHaveCount(0);
        }
        await select("s&box Ghost Collisions");
        await expect(
            panel.getByRole("region", { name: "Ghost launches", exact: true }),
        ).toBeVisible();
        for (const label of ["Walk speed X, in/s", "Walk speed Z, in/s"]) {
            const slider = panel.getByRole("slider", { name: label, exact: true });
            await slider.scrollIntoViewIfNeeded();
            await pointerControl(page, slider);
            await bodyHasFocus(page);
        }
        const reset = panel.getByRole("button", { name: "Reset Counters", exact: true });
        await reset.scrollIntoViewIfNeeded();
        await pointerControl(page, reset);
        await bodyHasFocus(page);
        await expect(panel.getByText("Vertical velocity", { exact: true })).toHaveCount(0);
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        await select("Restitution Overshoot");
        await expect(
            panel.getByRole("region", { name: "Bounce height", exact: true }),
        ).toBeVisible();
        await expect(panel.getByText("Drop height, m", { exact: true })).toBeVisible();
        await expect(panel.getByText("Max bounce, m", { exact: true })).toBeVisible();
        await expect(panel.getByText("At or below drop", { exact: true })).toBeVisible();
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
    });
