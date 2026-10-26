import { holdFrames, pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);
for (const name of [
    "Large Pyramid",
    "Wide Pyramid",
    "Many Pyramids",
    "Falling Boxes",
    "Joint Grid",
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
        // Let the orbit camera settle on the selected scene before capturing its view.
        await frames.run(12);
        await capture(page, name.replaceAll(" ", "-").toLowerCase(), frames);
        expect(errors).toEqual([]);
    });
}
