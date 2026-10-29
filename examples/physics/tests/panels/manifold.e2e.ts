import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Manifold cache controls and fixed contact rows fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        for (const [name, count] of [
            ["Sphere vs Sphere", 1],
            ["Capsule vs Sphere", 1],
            ["Hull vs Sphere", 1],
            ["Triangle vs Sphere", 1],
            ["Capsule vs Capsule", 2],
            ["Capsule vs Hull", 2],
            ["Triangle vs Capsule", 2],
            ["Hull vs Hull", 4],
            ["Triangle vs Hull", 7],
        ] as const) {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Manifold", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
            await page.evaluate(() =>
                (window as unknown as { controlsKey(): number }).controlsKey(),
            );
            const panel = page.locator(".controls"),
                results = panel.getByRole("region", { name: "Contact points", exact: true }),
                feature = panel.getByLabel("Feature", { exact: true });
            await expect(results.locator("output")).toHaveCount(count + 1);
            await expect(feature).toBeDisabled();
            const before = await panel.boundingBox();
            await pointerControl(
                page,
                panel.getByRole("checkbox", { name: "Use cache", exact: true }),
            );
            await bodyHasFocus(page);
            await expect(feature).toBeEnabled();
            await feature.selectOption("2");
            await bodyHasFocus(page);
            await pointerControl(
                page,
                panel.getByRole("checkbox", { name: "Use cache", exact: true }),
            );
            await bodyHasFocus(page);
            await expect(feature).toBeDisabled();
            await expect(feature).toHaveValue("2");
            expect(await panel.boundingBox()).toEqual(before);
            expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
            expect(
                await panel
                    .locator("label,output")
                    .evaluateAll((elements) =>
                        elements.every((el) => el.scrollWidth <= el.clientWidth),
                    ),
            ).toBe(true);
        }
    });
