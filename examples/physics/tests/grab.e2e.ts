import { expect, test } from "playwright/test";

test("a Box Stack cube follows the cursor grab and release removes the mouse and joint", async ({
    page,
}) => {
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    const result = await page.evaluate(() => (window as any).drag());
    expect(result.picked).toBe(result.body);
    expect(result.projected[0] - result.pixel[0]).toBeGreaterThan(60);
    expect(result.projected[0] - result.pixel[0]).toBeLessThan(110);
    expect(Math.abs(result.projected[1] - result.pixel[1])).toBeLessThan(15);
    expect(result.mouse).toBe(0);
    expect(result.joint).toBe(0);
});
