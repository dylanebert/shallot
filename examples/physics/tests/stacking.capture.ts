import { test } from "playwright/test";
import { stackingScenes } from "../src/stacking";
import { capture } from "./capture";

for (const id of Object.keys(stackingScenes)) {
    test(`${id} stacking look review`, async ({ page }) => {
        page.on("pageerror", (error) => {
            throw error;
        });
        await page.goto(`/?scene=${id}`);
        await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
        await capture(page, `${id}-default`);
        await page.waitForTimeout(10000);
        await capture(page, `${id}-settled`);
    });
}
