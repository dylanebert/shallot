import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`World presets and read-only rows fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        const select = async (name: string) => {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "World", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
        };
        const panel = page.locator(".controls");
        await select("Far Stack");
        await expect(panel.getByLabel("Offset, km", { exact: true })).toHaveText("0.0");
        for (const [name, value] of [
            ["10km", "10.0"],
            ["100km", "100.0"],
            ["1000km", "1000.0"],
            ["10000km", "10000.0"],
            ["origin", "0.0"],
        ]) {
            await pointerControl(page, panel.getByRole("button", { name: name!, exact: true }));
            await expect(panel.getByLabel("Offset, km", { exact: true })).toHaveText(value!);
            await expect(panel.getByLabel("Top height, m", { exact: true })).toHaveText(
                /^\d+\.\d{4}$/,
            );
            await bodyHasFocus(page);
        }
        for (const [name, value] of [
            ["Far Pyramid", "10000"],
            ["Far Ragdolls", "1000"],
            ["Far Mesh Drop", "1000"],
        ]) {
            await select(name!);
            await expect(panel.getByLabel("Offset, km", { exact: true })).toHaveText(value!);
            await expect(panel.getByRole("slider")).toHaveCount(0);
            await expect(panel.getByText(/double precision/i)).toHaveCount(0);
            expect(
                await panel
                    .locator("span, output")
                    .evaluateAll((elements) =>
                        elements.every((el) => el.scrollWidth <= el.clientWidth),
                    ),
            ).toBe(true);
        }
        await expect(panel.getByLabel("Result", { exact: true })).toHaveCount(0);
        await page.evaluate(() => {
            const step = (window as unknown as { controlsKey(): number }).controlsKey;
            for (let i = 0; i < 300; i++) step();
        });
        await expect(panel.getByLabel("Result", { exact: true })).toHaveText("failed!");
    });
