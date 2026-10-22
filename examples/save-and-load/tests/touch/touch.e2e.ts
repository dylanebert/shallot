import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";

test.use(pipelineWarnings);

for (const touch of [false, true]) {
    test(`${touch ? "touch" : "desktop"} instructions name the input in use`, async ({
        browser,
    }, testInfo) => {
        const context = await browser.newContext({
            viewport: { width: 390, height: 844 },
            hasTouch: touch,
        });
        const page = await context.newPage();
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto("http://127.0.0.1:4190/tests/touch.html");
        await page.waitForFunction(() => "hint" in window);
        const hint = (value: string) =>
            page.evaluate(
                (value) => (window as unknown as { hint(value: string): boolean }).hint(value),
                value,
            );
        await expect.poll(() => hint("Click for a coin")).toBe(true);
        if (touch) await page.touchscreen.tap(30, 500);
        await expect.poll(() => hint(touch ? "Tap for a coin" : "Click for a coin")).toBe(true);
        for (const key of await page.locator("kbd").all()) {
            if (touch) await expect(key).toBeHidden();
            else await expect(key).toBeVisible();
        }
        await page.screenshot({ path: testInfo.outputPath("save-and-load-phone.png") });
        expect(errors).toEqual([]);
        await context.close();
    });
}
