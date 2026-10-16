import { expect, type Page, test } from "playwright/test";

async function energy(page: Page): Promise<number[]> {
    return page.evaluate(async () => {
        const p = window as any;
        const sum = [0, 0];
        for (let i = 0; i < 20; i++) {
            p.advance();
            await new Promise((resolve) => setTimeout(resolve, 25));
            const e = p.energy();
            sum[0]! += e[0];
            sum[1]! += e[1];
        }
        return sum.map((x) => x / 20);
    });
}
test("the built yard serves its music and shared interaction HUD", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".reticle")).toBeVisible();
    expect((await page.request.get("/Project_Utopia.ogg")).ok()).toBe(true);
});
test("spatial music pans, fades, pauses and resumes; the bell finishes once", async ({ page }) => {
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    await page.getByRole("button", { name: "Start audio" }).click();
    await page.evaluate(() => {
        const p = window as any;
        p.freeze();
        p.pose(3, 1.4, -2);
    });
    const left = await energy(page);
    expect(left[0]!).toBeGreaterThan(0.000001);
    expect(left[0]!).toBeGreaterThan(left[1]! * 1.2);
    await page.evaluate(() => (window as any).pose(18, 1.4, -2));
    const far = await energy(page);
    expect(far[0]!).toBeLessThan(left[0]! * 0.3);
    expect(far[1]!).toBeLessThan(left[1]! * 0.3);
    await page.evaluate(() => {
        const p = window as any;
        p.aim();
        p.key(true);
        p.key(false);
    });
    expect(await page.evaluate(() => (window as any).state().paused)).toBe(1);
    await energy(page);
    const silent = await energy(page);
    expect(Math.max(...silent)).toBeLessThan(1e-9);
    await page.evaluate(() => {
        const p = window as any;
        p.key(true);
        p.key(false);
    });
    expect(await page.evaluate(() => (window as any).state().paused)).toBe(0);
    const resumed = await energy(page);
    expect(Math.min(...resumed)).toBeGreaterThan(0.000001);
    await page.evaluate(() => (window as any).bell());
    expect(await page.evaluate(() => (window as any).state().once)).toBe(1);
    await expect
        .poll(async () => {
            await energy(page);
            return page.evaluate(() => (window as any).state().once);
        })
        .toBe(0);
});
