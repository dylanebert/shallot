import { test } from "playwright/test";
import { capture } from "./capture";

for (const width of [1280, 390]) {
    for (const id of ["restitution", "staticInvoke", "wind"]) {
        test(`${id} Shapes look review at ${width}`, async ({ page }) => {
            await page.setViewportSize({ width, height: width === 390 ? 844 : 720 });
            await page.goto(`/?scene=${id}`);
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            await page.waitForTimeout(1000);
            await capture(page, `${id}-${width}-default`);
        });
    }
}
