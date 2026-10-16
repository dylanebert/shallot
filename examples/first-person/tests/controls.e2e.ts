import { expect, test } from "playwright/test";

test("spawn teaches controls in world pixels without an overlay controls panel", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4176/tests/stepped.html");
    await page.waitForFunction(() => (window as Window & { ready?: boolean }).ready);
    await expect(page.locator("body")).not.toContainText(/WASD|SPACE|Move|Jump|Look/);
    await expect(page.locator("[data-recipe-controls], [data-control-row]")).toHaveCount(0);
    const ink = await page.evaluate(() =>
        (window as unknown as Window & { controlsInk: () => Promise<number> }).controlsInk(),
    );
    expect(
        ink,
        "hiding the world controls removes visible ink from the spawn frame",
    ).toBeGreaterThan(500);
});
