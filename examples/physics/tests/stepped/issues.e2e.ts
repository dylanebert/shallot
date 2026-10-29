import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Issues scenes and Crash's action fit at ${width}`, async ({ page }) => {
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
            "s&box mover",
            "Slide Twist Off Center Shape",
        ]) {
            await select(name);
            await expect(panel).toHaveCount(0);
        }
    });
