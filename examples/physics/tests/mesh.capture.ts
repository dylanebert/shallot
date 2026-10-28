import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);
for (const scene of [
    "voxel",
    "hollowBox",
    "gridMesh",
    "bigBoxMesh",
    "boxMesh",
    "meshReflection",
    "meshHeightField",
    "meshViewer",
    "meshCreation",
]) {
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
            const hasShape = (await shape.count()) > 0;
            if (hasShape) {
                await shape.selectOption("0");
                await page.getByRole("slider", { name: "Scale X", exact: true }).fill("-1.5");
            } else if (scene === "meshReflection") {
                await page
                    .getByRole("combobox", { name: "Scale Y", exact: true })
                    .selectOption("-1");
            } else if (scene === "meshHeightField") {
                await page.getByRole("slider", { name: "radius", exact: true }).fill("0");
            } else if (scene === "meshViewer") {
                await page.getByRole("slider", { name: "index", exact: true }).fill("2");
                await page.getByRole("slider", { name: "draw level", exact: true }).fill("2");
            }
            if (
                hasShape ||
                scene === "meshReflection" ||
                scene === "meshHeightField" ||
                scene === "meshViewer"
            ) {
                await page.waitForTimeout(500);
                await capture(page, `${scene}-${size}-controls`);
                await page.getByText("Physics", { exact: true }).click();
                await capture(page, `${scene}-${size}-scene`);
            }
        }
    });
}
