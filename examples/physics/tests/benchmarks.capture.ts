import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);
for (const name of [
    "Height Field",
    "Hull",
    "Large Pyramid",
    "Wide Pyramid",
    "Many Pyramids",
    "Falling Boxes",
    "Joint Grid",
    "Rain",
    "Junkyard",
    "Washer",
    "Convex Pile",
    "Candy Cups",
    "Falling Trees",
    "Chains",
    "Sensor",
    "Explosion",
    "Destruction",
]) {
    test(`${name} look review`, async ({ page }) => {
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        const session = await page.context().newCDPSession(page);
        await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
        const frames = await holdFrames(page, "[aria-label='Scene navigation']");
        await page.goto("/");
        const picker = page.getByRole("button", { name: "Scene", exact: true });
        await expect(picker).toBeEnabled();
        await picker.click();
        await page
            .getByRole("region", { name: "Benchmark", exact: true })
            .getByRole("button", { name, exact: true })
            .click();
        await expect(picker).toHaveText(name);
        await expect(picker).toBeEnabled();
        // Advance the camera transition without running an unbounded presentation loop.
        await frames.run(name === "Sensor" ? 40 : 4);
        if (name === "Height Field") {
            await expect(page.locator(".input-help")).toContainText("count = 14278");
            await page.getByLabel("Radius", { exact: true }).fill("0");
            await frames.run(2);
            await capture(page, "height-field-rays", frames);
        }
        if (name === "Hull")
            await expect(page.locator(".input-help")).toContainText("trials = 2000");
        if (name === "Sensor") {
            await expect(page.locator(".input-help")).toContainText("max begin touch events =");
        }
        await capture(page, name.replaceAll(" ", "-").toLowerCase(), frames);
        if (name === "Explosion") {
            await page.getByLabel("Magnitude", { exact: true }).fill("1500");
            await page.getByRole("button", { name: "Explode", exact: true }).click();
            await frames.run(4);
            await capture(page, "explosion-fired", frames);
        }
        if (name === "Falling Trees") {
            for (const size of [50, 25, 100]) {
                await page.getByRole("radio", { name: `${size}cm`, exact: true }).check();
                await expect(
                    page.getByRole("radio", { name: `${size}cm`, exact: true }),
                ).toBeChecked();
                await frames.run(1);
                await capture(page, `falling-trees-${size}cm`, frames);
            }
            await page.setViewportSize({ width: 390, height: 844 });
            await frames.run(2);
            const panel = page.getByRole("radiogroup", { name: "Grid size", exact: true });
            const bounds = await panel.boundingBox();
            expect(bounds).not.toBeNull();
            expect(bounds!.x).toBeGreaterThanOrEqual(0);
            expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
            await capture(page, "falling-trees-phone", frames);
        }
        expect(errors).toEqual([]);
    });
}
