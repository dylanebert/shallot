import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { categories } from "../src/scenes";
import { scenePixels } from "./scenes";

test.use({
    ...pipelineWarnings,
    viewport: { width: 1280, height: 720 },
    hasTouch: true,
    isMobile: true,
    deviceScaleFactor: 1,
});

test("catalog switches draw on the same canvas with desktop clicks and 390x844 touch", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
    });
    const frames = await holdFrames(page, "[aria-label='Scene']");
    await page.goto("/");
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await picker.waitFor();
    await expect(picker).toHaveText("CesiumMilkTruck");
    const canvas = await page.locator("canvas").elementHandle();
    // The truck's green texture, not its silhouette or page chrome.
    await expect
        .poll(async () => {
            expect(errors).toEqual([]);
            await frames.run();
            return scenePixels(page);
        })
        .toBeGreaterThan(100);

    for (const phone of [false, true]) {
        if (phone) await page.setViewportSize({ width: 390, height: 844 });
        expect(await page.evaluate(() => [innerWidth, innerHeight])).toEqual(
            phone ? [390, 844] : [1280, 720],
        );
        if (phone) await picker.tap();
        else await picker.click();
        for (const name of categories.map((category) => category.name))
            await expect(page.getByRole("region", { name, exact: true })).toBeVisible();
        const name = phone ? "CesiumMilkTruck" : "TextureCoordinateTest";
        const choice = page
            .getByRole("region", { name: "textures", exact: true })
            .getByRole("button", { name, exact: true });
        if (phone) await choice.tap();
        else await choice.click();
        await expect(picker).toHaveText(name);
        await expect(picker).toHaveAttribute("aria-expanded", "false");
        // The UV test's blue patches and the truck's green body.
        await expect
            .poll(async () => {
                expect(errors).toEqual([]);
                await frames.run();
                return scenePixels(page, !phone);
            })
            .toBeGreaterThan(100);
        expect(
            await page
                .locator("canvas")
                .evaluate((current, original) => current === original, canvas),
        ).toBe(true);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
            true,
        );
        await expect(page.getByRole("slider")).toHaveCount(0);
    }
    for (const category of categories) {
        for (const scene of category.scenes) {
            await picker.click();
            await page
                .getByRole("region", { name: category.name, exact: true })
                .getByRole("button", { name: scene.id, exact: true })
                .click();
            await expect(picker).toHaveText(scene.id);
            await frames.run(2);
        }
    }
    expect(errors).toEqual([]);
});
