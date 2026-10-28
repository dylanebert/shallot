import { test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390]) {
    for (const id of [
        "bodyType",
        "weeble",
        "disable",
        "gyroscopicPrecession",
        "classRing",
        "cast",
    ]) {
        test(`${id} Bodies look review at ${width}`, async ({ page }) => {
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            await page.goto(`/?scene=${id}`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            if (id === "gyroscopicPrecession")
                await page.getByLabel("Expected average", { exact: true }).waitFor();
            else await page.waitForTimeout(1000);
            await capture(page, `${id}-${width}-default`);
            if (id === "bodyType") {
                await page.getByRole("combobox", { name: "Type", exact: true }).selectOption("1");
                await capture(page, `${id}-${width}-kinematic`);
            } else if (id === "weeble") {
                await page.getByRole("button", { name: "Teleport", exact: true }).click();
                await capture(page, `${id}-${width}-teleport`);
            } else if (id === "disable") {
                await page.getByRole("checkbox", { name: "Enable Link", exact: true }).uncheck();
                await capture(page, `${id}-${width}-disabled`);
            }
        });
    }
}
