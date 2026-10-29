import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Tree Benchmark controls and result rows fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(
            page,
            page
                .getByRole("region", { name: "Tree", exact: true })
                .getByRole("button", { name: "Benchmark", exact: true }),
        );
        const panel = page.locator(".controls"),
            bounds = await panel.boundingBox();
        await expect(panel.getByLabel("Leaves", { exact: true })).toHaveText("288");
        await expect(
            panel
                .getByRole("region", { name: "Tree statistics and cost", exact: true })
                .locator("output"),
        ).toHaveCount(10);
        for (const name of ["Ray Cast", "Overlap", "Closet Point"]) {
            await pointerControl(page, panel.getByLabel(name, { exact: true }));
            await expect(panel.getByLabel(name, { exact: true })).toBeChecked();
        }
        await pointerControl(page, panel.getByRole("button", { name: "Save", exact: true }));
        await panel.getByLabel("File", { exact: true }).selectOption("2");
        await bodyHasFocus(page);
        await expect(panel.getByLabel("Leaves", { exact: true })).toHaveText("268");
        await pointerControl(page, panel.getByRole("button", { name: "Load", exact: true }));
        await expect(panel.getByLabel("Leaves", { exact: true })).toHaveText("288");
        const level = panel.getByLabel("Level", { exact: true });
        await expect(level).toHaveAttribute("max", /^[1-9][0-9]*$/);
        await level.fill("0");
        await pointerControl(page, panel.getByRole("button", { name: "Top Down", exact: true }));
        await pointerControl(page, panel.getByRole("button", { name: "Profile", exact: true }));
        const after = await panel.boundingBox();
        expect(after!.x).toBe(bounds!.x);
        expect(after!.y).toBe(bounds!.y);
        expect(after!.width).toBe(bounds!.width);
        expect(after!.height).toBe(bounds!.height);
        expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        expect(
            await panel
                .locator("label, span, output")
                .evaluateAll((elements) =>
                    elements.every((el) => el.scrollWidth <= el.clientWidth),
                ),
        ).toBe(true);
    });
