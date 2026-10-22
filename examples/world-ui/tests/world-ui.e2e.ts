import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test("the copied production page presents the delivery scene", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await expect(page.locator("canvas")).toBeVisible();
    await expect
        .poll(() =>
            page.locator("canvas").evaluate((canvas) => (canvas as HTMLCanvasElement).width),
        )
        .toBeGreaterThan(1);
    expect(errors).toEqual([]);
});
