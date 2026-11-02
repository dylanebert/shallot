import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { bodyHasFocus, pointerControl } from "./pointer";

export function manifoldPanelTests(stepped: boolean) {
    test.use(pipelineWarnings);
    for (const width of [1280, 390])
        test(`Manifold drawing and cache controls with fixed contact rows fit at ${width}`, async ({
            page,
        }) => {
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            const frames = stepped
                ? null
                : await holdFrames(page, "[aria-label='Scene navigation']");
            await page.goto(stepped ? "/tests/controls.html" : "/");
            if (stepped) await page.waitForFunction(() => "controlsKey" in window);
            else {
                await expect(
                    page.getByRole("button", { name: "Scene", exact: true }),
                ).toBeEnabled();
                // runApp mounts the enabled UI beneath its loading screen.
                await expect(page.locator("[data-shallot-loading]")).toHaveCount(0);
            }
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
                await pointerControl(
                    page,
                    page.getByRole("button", { name: "Scene", exact: true }),
                );
                await pointerControl(
                    page,
                    page
                        .getByRole("region", { name: "Manifold", exact: true })
                        .getByRole("button", { name, exact: true }),
                );
                if (frames) await frames.run();
                else
                    await page.evaluate(() =>
                        (window as unknown as { controlsKey(): number }).controlsKey(),
                    );
                const panel = page.locator(".controls"),
                    results = panel.getByRole("region", { name: "Contact points", exact: true }),
                    feature = panel.getByLabel("Feature", { exact: true });
                await expect(results.locator("output")).toHaveCount(count + 1);
                await expect(feature).toBeDisabled();
                const before = await panel.boundingBox();
                const points = panel.getByRole("checkbox", { name: "Draw Points", exact: true });
                if (name.startsWith("Triangle")) await expect(points).toHaveCount(0);
                else {
                    await expect(points).toBeChecked();
                    await pointerControl(page, points);
                    await expect(points).not.toBeChecked();
                    await bodyHasFocus(page);
                    await pointerControl(page, points);
                    await expect(points).toBeChecked();
                    await bodyHasFocus(page);
                }
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
    for (const width of [1280, 390])
        test(`Manifold diagnostic controls and readouts work at ${width}`, async ({ page }) => {
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            const frames = stepped
                ? null
                : await holdFrames(page, "[aria-label='Scene navigation']");
            await page.goto(stepped ? "/tests/controls.html" : "/");
            if (stepped) await page.waitForFunction(() => "controlsKey" in window);
            else {
                await expect(
                    page.getByRole("button", { name: "Scene", exact: true }),
                ).toBeEnabled();
                await expect(page.locator("[data-shallot-loading]")).toHaveCount(0);
            }
            const advance = async () => {
                if (frames) await frames.run();
                else
                    await page.evaluate(() =>
                        (window as unknown as { controlsKey(): number }).controlsKey(),
                    );
            };
            const select = async (name: string) => {
                await pointerControl(
                    page,
                    page.getByRole("button", { name: "Scene", exact: true }),
                );
                await pointerControl(
                    page,
                    page
                        .getByRole("region", { name: "Manifold", exact: true })
                        .getByRole("button", { name, exact: true }),
                );
                await advance();
            };
            await select("Hull Culling");
            const panel = page.locator(".controls"),
                feature = panel.getByLabel("Feature", { exact: true }),
                height = panel.getByLabel("Height", { exact: true });
            await expect(feature).toBeDisabled();
            await pointerControl(
                page,
                panel.getByRole("checkbox", { name: "Use cache", exact: true }),
            );
            await expect(feature).toBeEnabled();
            await feature.selectOption("1");
            await bodyHasFocus(page);
            await expect(height).toBeDisabled();
            await pointerControl(
                page,
                panel.getByRole("checkbox", { name: "Cylinders", exact: true }),
            );
            await expect(height).toBeEnabled();
            await pointerControl(page, height);
            expect(Number(await height.inputValue())).toBeGreaterThan(2);
            await panel.getByLabel("Edge bound", { exact: true }).selectOption("0");
            await bodyHasFocus(page);
            await pointerControl(
                page,
                panel.getByRole("checkbox", { name: "Culled features", exact: true }),
            );
            await advance();
            await select("Edge Axis Wedge");
            await expect(panel.getByLabel("Winner", { exact: true })).toHaveText("edge");
            await expect(panel.getByLabel("Search", { exact: true })).toContainText("found /");
            await panel.getByLabel("Hull", { exact: true }).selectOption("2");
            await advance();
            await pointerControl(
                page,
                panel.getByRole("button", { name: "Find Example", exact: true }),
            );
            await advance();
            await bodyHasFocus(page);
            await expect(panel.getByLabel("Winner", { exact: true })).toHaveText("edge");
            await expect(panel.getByLabel("Search", { exact: true })).toContainText("found /");
            expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
        });
}
