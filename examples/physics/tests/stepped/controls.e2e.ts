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
    test(`Mesh signs and terrain bounds fit their fields at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(page, page.getByRole("button", { name: "Reflection", exact: true }));
        const panel = page.locator(".controls");
        for (const axis of ["X", "Y", "Z"]) {
            const select = panel.getByLabel(`Scale ${axis}`, { exact: true });
            expect(await select.locator("option").allTextContents()).toEqual(["Neg", "Pos"]);
            for (const value of ["-1", "1"]) {
                await select.selectOption(value);
                await expect(select).toHaveValue(value);
                await bodyHasFocus(page);
            }
        }
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        const terrain = page
            .getByRole("region", { name: "Mesh", exact: true })
            .getByRole("button", { name: "Height Field", exact: true });
        await pointerControl(page, terrain);
        const columns = panel.getByLabel("columns", { exact: true });
        await columns.scrollIntoViewIfNeeded();
        const columnBox = (await columns.boundingBox())!;
        await page.mouse.move(
            columnBox.x + columnBox.width / 2,
            columnBox.y + columnBox.height / 2,
        );
        await page.mouse.down();
        await page.mouse.move(columnBox.x, columnBox.y + columnBox.height / 2, { steps: 8 });
        await page.mouse.up();
        await expect(columns).toHaveValue("1");
        await bodyHasFocus(page);
        const ray = panel.getByLabel("ray x", { exact: true });
        await expect(ray).toHaveAttribute("min", "-0.4");
        await expect(ray).toHaveAttribute("max", "0.4");
        const radius = panel.getByLabel("radius", { exact: true });
        await radius.scrollIntoViewIfNeeded();
        const radiusBox = (await radius.boundingBox())!;
        await page.mouse.move(
            radiusBox.x + radiusBox.width / 2,
            radiusBox.y + radiusBox.height / 2,
        );
        await page.mouse.down();
        await page.mouse.move(radiusBox.x + radiusBox.width, radiusBox.y + radiusBox.height / 2, {
            steps: 8,
        });
        await page.mouse.up();
        await expect(radius).toHaveValue("1");
        await bodyHasFocus(page);
        await expect(ray).toHaveAttribute("min", "-2");
        await expect(ray).toHaveAttribute("max", "2");
        expect(await panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
            true,
        );
    });

    test(`Shapes choices fit their fixed fields at ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        for (const scene of ["Restitution", "Static Invoke", "Wind"]) {
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(page, page.getByRole("button", { name: scene, exact: true }));
            const panel = page.locator(".controls");
            expect(
                await panel.evaluate((element) => element.scrollWidth <= element.clientWidth),
            ).toBe(true);
            if (scene === "Restitution") {
                for (const name of ["Box", "Sphere"])
                    await pointerControl(page, panel.getByRole("radio", { name, exact: true }));
            } else {
                const select = panel.getByRole("combobox");
                const bounds = (await panel.boundingBox())!;
                const field = (await select.boundingBox())!;
                expect(field.x + field.width).toBeLessThanOrEqual(bounds.x + bounds.width - 16);
                for (const value of scene === "Wind" ? ["0", "1", "2"] : ["1", "0"]) {
                    await select.selectOption(value);
                    await expect(select).toHaveValue(value);
                    await bodyHasFocus(page);
                }
            }
            await gameKeyMovesCamera(page);
        }
    });

    test(`Static Invoke swaps live actions without moving its panel at ${width}`, async ({
        page,
    }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(
            page,
            page.getByRole("button", { name: "Static Invoke", exact: true }),
        );
        const panel = page.locator(".controls");
        const create = panel.getByRole("button", { name: "Create", exact: true });
        const destroy = panel.getByRole("button", { name: "Destroy", exact: true });
        const bounds = await panel.boundingBox();
        await expect(create).toBeEnabled();
        await expect(destroy).toBeDisabled();
        await page.evaluate(() => {
            for (let i = 0; i < 20; i++)
                (window as unknown as { controlsKey(): number }).controlsKey();
        });
        await expect(create).toBeDisabled();
        await expect(destroy).toBeEnabled();
        expect(await panel.boundingBox()).toEqual(bounds);
        for (let i = 0; i < 2; i++) {
            await pointerControl(page, destroy);
            await expect(create).toBeEnabled();
            await expect(destroy).toBeDisabled();
            expect(await panel.boundingBox()).toEqual(bounds);
            await pointerControl(page, create);
            await expect(create).toBeDisabled();
            await expect(destroy).toBeEnabled();
            expect(await panel.boundingBox()).toEqual(bounds);
        }
    });

    test(`Sensor Hits keeps actions and read-only touch counts in separate sections at ${width}`, async ({
        page,
    }) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
        await page.goto("/tests/controls.html");
        await page.waitForFunction(() => "controlsKey" in window);
        await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
        await pointerControl(page, page.getByRole("button", { name: "Sensor Hits", exact: true }));
        const panel = page.locator(".controls");
        const bounds = await panel.boundingBox();
        const actions = panel.getByRole("region", { name: "Simulation", exact: true });
        const counts = panel.getByRole("region", { name: "Touch counts", exact: true });
        await expect(actions.getByRole("checkbox", { name: "Bullet", exact: true })).toBeChecked();
        for (const label of ["Begin touch", "End touch"])
            await expect(counts.getByLabel(label, { exact: true })).toHaveText("0");
        await expect(counts.locator("input, button")).toHaveCount(0);
        await pointerControl(page, actions.getByRole("checkbox", { name: "Bullet", exact: true }));
        await pointerControl(page, actions.getByRole("button", { name: "Launch", exact: true }));
        expect(await panel.boundingBox()).toEqual(bounds);
        expect(await panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
            true,
        );
        await gameKeyMovesCamera(page);
    });

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
