import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { categories } from "../src/scenes";
import { scenePixels } from "./scenes";

test.use({ ...pipelineWarnings, deviceScaleFactor: 1 });

for (const [width, height] of [
    [1280, 720],
    [1512, 982],
    [390, 844],
]) {
    test.describe(`${width}x${height}`, () => {
        const phone = width === 390;
        test.use({ viewport: { width, height }, hasTouch: phone, isMobile: phone });
        test("compact categories switch every model on the same canvas", async ({ page }) => {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            page.on("console", (message) => {
                if (message.type() === "error") errors.push(message.text());
            });
            const frames = await holdFrames(page, "[aria-label='Scene']");
            await page.goto("/");
            const picker = page.getByRole("button", { name: "Scene", exact: true });
            await expect(picker).toHaveText("CesiumMilkTruck");
            const canvas = await page.locator("canvas").elementHandle();
            await expect
                .poll(async () => {
                    expect(errors).toEqual([]);
                    await frames.run();
                    return scenePixels(page);
                })
                .toBeGreaterThan(100);

            if (phone) await picker.tap();
            else await picker.click();
            await expect
                .poll(() =>
                    page
                        .locator(".categories")
                        .evaluate((node) => Number(getComputedStyle(node).columnCount)),
                )
                .toBe(phone ? 1 : categories.length);
            const sections = await page.locator(".categories section").evaluateAll((nodes) =>
                nodes.map((node) => ({
                    fragments: node.getClientRects().length,
                    x: Math.round(node.getBoundingClientRect().x),
                    breakInside: getComputedStyle(node).breakInside,
                })),
            );
            expect(
                sections.every(
                    (section) => section.fragments === 1 && section.breakInside === "avoid",
                ),
            ).toBe(true);
            expect(new Set(sections.map((section) => section.x)).size).toBe(
                phone ? 1 : categories.length,
            );
            if (phone) await picker.tap();
            else await picker.click();

            for (const category of categories) {
                for (const scene of category.scenes) {
                    if (phone) await picker.tap();
                    else await picker.click();
                    const choice = page
                        .getByRole("region", { name: category.name, exact: true })
                        .getByRole("button", { name: scene.id, exact: true });
                    if (phone) await choice.tap();
                    else await choice.click();
                    await expect(picker).toHaveText(scene.id);
                    await expect(picker).toHaveAttribute("aria-expanded", "false");
                    await frames.run(2);
                    if (scene.id === "TextureCoordinateTest") {
                        await expect.poll(() => scenePixels(page, true)).toBeGreaterThan(100);
                    }
                }
            }
            expect(
                await page
                    .locator("canvas")
                    .evaluate((current, original) => current === original, canvas),
            ).toBe(true);
            expect(
                await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
            ).toBe(true);
            await expect(page.getByRole("slider")).toHaveCount(0);
            expect(errors).toEqual([]);
        });
    });
}
