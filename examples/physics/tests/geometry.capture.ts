import { test } from "playwright/test";
import { geometryNames } from "../src/geometry";
import { capture } from "./capture";

for (const width of [1280, 390])
    for (const id of geometryNames) {
        test(`${id} Geometry look review at ${width}`, async ({ page }) => {
            page.on("pageerror", (error) => {
                throw error;
            });
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            await page.goto(`/?scene=${id}`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            await page.waitForTimeout(1000);
            await capture(page, `${id}-${width}-default`);
        });
    }
