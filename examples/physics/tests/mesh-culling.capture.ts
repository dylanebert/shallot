import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);

for (const scene of ["sensorHits", "conveyorMesh", "heightField", "hollowBox"]) {
    test(`${scene} single-sided mesh review`, async ({ page }) => {
        page.on("pageerror", (error) => {
            throw error;
        });
        for (const [size, width, height] of [
            ["desktop", 1280, 720],
            ["phone", 390, 844],
        ] as const) {
            await page.setViewportSize({ width, height });
            await page.goto(`/?scene=${scene}`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            await page.waitForTimeout(1500);
            const panel = page.getByText("Physics", { exact: true });
            if (await panel.count()) await panel.click();
            await capture(page, `${scene}-single-sided-${size}`);
        }
    });
}
