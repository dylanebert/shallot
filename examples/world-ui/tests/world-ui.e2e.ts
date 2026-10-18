import { expect, test } from "playwright/test";

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
