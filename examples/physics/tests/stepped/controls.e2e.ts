import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { categories } from "../../src/scenes";
import {
    bodyHasFocus,
    dragRange,
    gameKeyMovesCamera,
    pointerControl,
    tabSkipsChrome,
} from "../pointer";

const samples = categories.flatMap((category) => category.scenes);
const beforeWheel = samples[samples.findIndex((sample) => sample.id === "wheel") - 1]!.name;

test.use(pipelineWarnings);

for (const width of [1280, 390]) {
    test(`Precession's averaged rates fit the label column at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(
            page,
            page.getByRole("button", { name: "Gyroscopic Precession", exact: true }),
        );
        await page.evaluate(() => {
            const step = (window as unknown as { controlsKey(): number }).controlsKey;
            for (let i = 0; i < 80; i++) step();
        });
        const section = page.getByRole("region", { name: "Precession, rad/s", exact: true });
        await expect(
            section.getByRole("heading", { name: "Precession, rad/s", exact: true }),
        ).toBeVisible();
        await expect(page.getByRole("heading", { name: "Simulation", exact: true })).toHaveCount(0);
        for (const label of ["Expected average", "Actual average"]) {
            await expect(section.getByLabel(label, { exact: true })).toBeVisible();
            const dimensions = await section
                .getByText(label, { exact: true })
                .evaluate((element) => ({
                    width: element.getBoundingClientRect().width,
                    fits: element.scrollWidth <= element.clientWidth,
                }));
            expect(dimensions.width).toBe(144);
            expect(dimensions.fits).toBe(true);
        }
        expect(
            await page
                .locator(".controls")
                .evaluate((element) => element.scrollWidth <= element.clientWidth),
        ).toBe(true);
    });

    test(`Body Type's native choice fits its field at ${width} and returns keys to the game`, async ({
        page,
    }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(page, page.getByRole("button", { name: "Body Type", exact: true }));
        const type = page.getByRole("combobox", { name: "Type", exact: true });
        const panel = page.locator(".controls");
        const box = (await type.boundingBox())!;
        const bounds = (await panel.boundingBox())!;
        expect(box.x + box.width).toBeLessThanOrEqual(bounds.x + bounds.width - 16);
        expect(await panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
            true,
        );
        await type.click();
        await page.keyboard.press("Escape");
        await bodyHasFocus(page);
        await type.selectOption("1");
        await expect(type).toHaveValue("1");
        await bodyHasFocus(page);
        await gameKeyMovesCamera(page);
        await tabSkipsChrome(page);
    });
}

test("pointer controls retain disabled nested settings and disclose the root without focus", async ({
    page,
}) => {
    await page.goto("/tests/controls.html");
    await page.waitForFunction(() => "controlsKey" in window);
    const picker = page.getByRole("button", { name: "Scene", exact: true });
    await pointerControl(page, picker);
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.getByRole("button", { name: "Wheel", exact: true }));
    await gameKeyMovesCamera(page);
    await expect(picker).toHaveText("Wheel");
    await pointerControl(page, page.getByRole("button", { name: "Previous scene", exact: true }));
    await expect(picker).toHaveText(beforeWheel);
    await gameKeyMovesCamera(page);
    await pointerControl(page, page.getByRole("button", { name: "Next scene", exact: true }));
    await expect(picker).toHaveText("Wheel");
    await gameKeyMovesCamera(page);
    const panel = page.locator(".controls");
    const steering = panel.getByRole("checkbox", { name: "Steering", exact: true });
    const limit = panel.getByRole("checkbox", { name: "Steering Limit", exact: true });
    const minimum = panel.getByRole("slider", { name: "Min Degrees", exact: true });
    await pointerControl(page, steering);
    await pointerControl(page, limit);
    await gameKeyMovesCamera(page);
    await dragRange(page, minimum);
    await gameKeyMovesCamera(page);
    const retained = await minimum.inputValue();
    const before = await panel.boundingBox();
    await pointerControl(page, steering);
    await expect(limit).toBeDisabled();
    await expect(limit).toBeChecked();
    await expect(minimum).toBeDisabled();
    await expect(minimum).toHaveValue(retained);
    expect(await panel.boundingBox()).toEqual(before);
    const title = panel.locator("summary");
    await pointerControl(page, title);
    await expect(steering).toBeHidden();
    await gameKeyMovesCamera(page);
    await pointerControl(page, title);
    await expect(steering).toBeVisible();
    await pointerControl(page, picker);
    await pointerControl(page, page.getByRole("button", { name: "Motor Joint", exact: true }));
    await expect(picker).toHaveText("Motor Joint");
    const speed = () =>
        page.evaluate(() => (window as unknown as { actionSpeed(): number }).actionSpeed());
    expect(await speed()).toBe(0);
    await pointerControl(page, page.getByRole("button", { name: "Apply impulse", exact: true }));
    expect(await speed()).toBeGreaterThan(0);
    await gameKeyMovesCamera(page);
    await tabSkipsChrome(page);
});
