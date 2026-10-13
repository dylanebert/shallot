import { expect, test } from "playwright/test";

test("StandardAudioPlugin plays a registered clip through AudioOutput's bus", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#status")).toHaveText("ready");
    await page.getByRole("button", { name: "Start audio" }).click();
    await expect
        .poll(() =>
            page.evaluate(
                () => (window as Window & { audioPeak?: () => number }).audioPeak?.() ?? 0,
            ),
        )
        .toBeGreaterThan(0.01);
});
