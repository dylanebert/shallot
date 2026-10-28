import { mkdir } from "node:fs/promises";
import { test } from "playwright/test";

for (const viewport of [
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
]) {
    test(`capture default ${viewport.width}`, async ({ browser }) => {
        const context = await browser.newContext({ viewport, hasTouch: viewport.width === 390 });
        const page = await context.newPage();
        try {
            await page.goto("http://127.0.0.1:4265/tests/stepped.html");
            await page.waitForFunction(() => (window as any).ready);
            await page.evaluate(() => document.fonts.ready);
            await page.evaluate(() => (window as any).captureScene());
            await mkdir(process.env.SHALLOT_CAPTURE_DIR!, { recursive: true });
            await page.screenshot({
                path: `${process.env.SHALLOT_CAPTURE_DIR}/default-${viewport.width}x${viewport.height}.png`,
            });
        } finally {
            await context.close();
        }
    });
}
