import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use({ ...pipelineWarnings, hasTouch: true });
for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`${viewport.width}x${viewport.height}: an auto-opened profiler overlaps no visible control`, async ({
        page,
    }) => {
        await page.setViewportSize(viewport);
        const session = await page.context().newCDPSession(page);
        await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
        const frames = await holdFrames(page, "[aria-label='Scene navigation']");
        await page.goto("/");
        const picker = page.getByRole("button", { name: "Scene", exact: true });
        await expect(picker).toBeEnabled();
        await picker.click();
        await page.getByRole("button", { name: "Falling Trees", exact: true }).click();
        await expect(
            page.getByRole("radiogroup", { name: "Grid size", exact: true }),
        ).toBeVisible();
        await frames.run(4);
        const controls = await page
            .locator(".ec button, .ec input, .ec select, .ec summary")
            .evaluateAll((nodes) =>
                nodes.map((node) => ({
                    name: node.textContent?.trim() || node.getAttribute("aria-label"),
                    box: node.getBoundingClientRect().toJSON(),
                })),
            );
        expect(controls.length).toBeGreaterThan(0);
        const overlaps = await page
            .locator("[data-shallot-profile]")
            .evaluateAll((profiles, controls) => {
                return profiles.flatMap((profile) => {
                    const a = profile.getBoundingClientRect();
                    return controls
                        .filter(({ box: b }) => {
                            return (
                                b.width > 0 &&
                                b.height > 0 &&
                                a.left < b.right &&
                                a.right > b.left &&
                                a.top < b.bottom &&
                                a.bottom > b.top
                            );
                        })
                        .map((control) => control.name);
                });
            }, controls);
        expect(overlaps).toEqual([]);
        await expect(page.locator("[data-shallot-profile]")).toHaveCount(
            viewport.width === 390 ? 0 : 1,
        );
        if (viewport.width === 1280) {
            await page.setViewportSize({ width: 390, height: 844 });
            await frames.run(1);
            await expect(page.locator("[data-shallot-profile]")).toHaveCount(0);
        }
        if (viewport.width === 390) {
            for (const size of [25, 100]) {
                const option = page.getByRole("radio", { name: `${size}cm`, exact: true });
                await option.tap();
                await expect(option).toBeChecked();
            }
            await expect(page.locator("[data-shallot-profile]")).toHaveCount(0);
        }
    });
}
