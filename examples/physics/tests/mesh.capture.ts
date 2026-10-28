import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);
for (const scene of ["voxel", "hollowBox", "gridMesh", "bigBoxMesh", "boxMesh"]) {
    test(`${scene} Mesh-stage look review`, async ({ page }) => {
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
            await capture(page, `${scene}-${size}-default`);
            const shape = page.getByRole("combobox", { name: "Shape", exact: true });
            if (await shape.count()) {
                await shape.selectOption("0");
                await page.getByRole("slider", { name: "Scale X", exact: true }).fill("-1.5");
                await page.waitForTimeout(500);
                await capture(page, `${scene}-${size}-controls`);
                await page.getByText("Physics", { exact: true }).click();
                await capture(page, `${scene}-${size}-scene`);
            }
        }
    });
}
