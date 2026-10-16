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
    await page.evaluate(() =>
        (window as Window & { pauseAudio?: (paused: boolean) => void }).pauseAudio!(true),
    );
    await expect
        .poll(() =>
            page.evaluate(() => (window as Window & { audioPeak?: () => number }).audioPeak!()),
        )
        .toBeLessThan(0.0001);
    await page.evaluate(() =>
        (window as Window & { pauseAudio?: (paused: boolean) => void }).pauseAudio!(false),
    );
    await expect
        .poll(() =>
            page.evaluate(() => (window as Window & { audioPeak?: () => number }).audioPeak!()),
        )
        .toBeGreaterThan(0.01);
});
