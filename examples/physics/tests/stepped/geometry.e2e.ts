import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Geometry choices and result sections fit at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        for (const [name, section, outputs] of [
            ["Box Hull", "Geometry", 12],
            ["Hull Reduction", "Reduced hull", 3],
            ["Hull Transform", "Hull geometry", 6],
            ["Capsule Mass", "Mass and inertia", 12],
        ] as const) {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page
                    .getByRole("region", { name: "Geometry", exact: true })
                    .getByRole("button", { name, exact: true }),
            );
            await page.evaluate(() =>
                (window as unknown as { controlsKey(): number }).controlsKey(),
            );
            const panel = page.locator(".controls");
            await expect(
                panel.getByRole("region", { name: section, exact: true }).locator("output"),
            ).toHaveCount(outputs);
            const slider = panel.getByRole("slider").first();
            await slider.fill((await slider.getAttribute("min")) ?? "0");
            await bodyHasFocus(page);
            if (name === "Hull Reduction") {
                await panel.getByLabel("Shape", { exact: true }).selectOption("0");
                await bodyHasFocus(page);
            }
            expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
            expect(
                await panel
                    .locator("label, output")
                    .evaluateAll((elements) =>
                        elements.every((el) => el.scrollWidth <= el.clientWidth),
                    ),
            ).toBe(true);
        }
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(
            page,
            page
                .getByRole("region", { name: "Geometry", exact: true })
                .getByRole("button", { name: "Hull", exact: true }),
        );
        await expect(page.locator(".controls")).toHaveCount(0);
    });
