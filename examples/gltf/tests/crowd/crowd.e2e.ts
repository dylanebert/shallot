import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { scenePixels } from "../scenes";

test.use({
    ...pipelineWarnings,
    viewport: { width: 390, height: 844 },
    hasTouch: true,
    isMobile: true,
    deviceScaleFactor: 1,
});

test("crowd controls switch posing modes and retain measured costs on a phone", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const frames = await holdFrames(page, "[aria-label='Scene']");
    await page.goto("/?scene=CesiumManCrowd");
    await expect(page.getByRole("button", { name: "Scene", exact: true })).toHaveText(
        "CesiumManCrowd",
    );
    const panel = page.getByLabel("CesiumManCrowd controls");
    for (const mode of ["live", "shared", "baked", "live"]) {
        await page.getByLabel("Pose", { exact: true }).selectOption(mode);
        await frames.run(3);
        await expect(page.getByLabel(`${mode} CPU`, { exact: true })).toHaveText(/^\d+\.\d{3}$/);
        await expect.poll(() => scenePixels(page)).toBeGreaterThan(100);
    }
    const box = await panel.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
        true,
    );
    await page.getByRole("button", { name: "Next scene" }).tap();
    await expect(panel).toHaveCount(0);
    expect(errors).toEqual([]);
});
