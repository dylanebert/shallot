import { expect, test } from "playwright/test";

test.use({ baseURL: "http://127.0.0.1:4196" });

test("phone fingers move, look and jump through the example's controls", async ({
    browser,
}, testInfo) => {
    const example = "view-model";
    const context = await browser.newContext({
        viewport: { width: 390, height: 844 },
        hasTouch: true,
    });
    const page = await context.newPage();
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/tests/touch.html");
    await page.waitForFunction(() => "sample" in window);
    const sample = () =>
        page.evaluate(() =>
            (
                window as unknown as {
                    sample(): { position: number[]; yaw: number; jumps: number; use: boolean };
                }
            ).sample(),
        );
    const session = await context.newCDPSession(page);
    let points: { id: number; x: number; y: number }[] = [];
    const send = (type: "touchStart" | "touchMove" | "touchEnd") =>
        session.send("Input.dispatchTouchEvent", { type, touchPoints: points });
    await expect(page.getByRole("button", { name: "Jump", exact: true })).toBeHidden();
    if (["interact", "sound"].includes(example)) {
        await expect(page.locator(".hint kbd")).toHaveText("E");
    }
    const before = await sample();
    points = [{ id: 1, x: 90, y: 620 }];
    await send("touchStart");
    points[0].y = 550;
    await send("touchMove");
    await expect
        .poll(async () => {
            const now = await sample();
            return Math.hypot(
                now.position[0] - before.position[0],
                now.position[2] - before.position[2],
            );
        })
        .toBeGreaterThan(0.1);
    points.push({ id: 2, x: 280, y: 450 });
    await send("touchStart");
    points[1].x = 330;
    await send("touchMove");
    await expect.poll(async () => (await sample()).yaw).not.toBe(before.yaw);
    points = [];
    await send("touchEnd");
    const jump = await page.getByRole("button", { name: "Jump", exact: true }).boundingBox();
    const grounded = await sample();
    const jumps = grounded.jumps;
    points.push({ id: 3, x: jump!.x + 32, y: jump!.y + 32 });
    await send("touchStart");
    await expect.poll(async () => (await sample()).jumps).toBeGreaterThan(jumps);
    await expect
        .poll(async () => (await sample()).position[1])
        .toBeGreaterThan(grounded.position[1] + 0.2);
    points.pop();
    await send("touchEnd");
    if (["interact", "sound"].includes(example)) {
        await page.evaluate(() => (window as unknown as { aimUse(): void }).aimUse());
        await expect
            .poll(() =>
                page.evaluate(() => (window as unknown as { useReady(): boolean }).useReady()),
            )
            .toBe(true);
        const useState = () =>
            page.evaluate(() => (window as unknown as { useState(): number }).useState());
        await expect(page.locator(".hint kbd")).toHaveText("Use");
        const previous = await useState();
        const use = await page.getByRole("button", { name: "Use", exact: true }).boundingBox();
        points.push({ id: 4, x: use!.x + 32, y: use!.y + 32 });
        await send("touchStart");
        await expect.poll(async () => (await sample()).use).toBe(true);
        await expect.poll(useState).not.toBe(previous);
        points.pop();
        await send("touchEnd");
        await expect.poll(async () => (await sample()).use).toBe(false);
    }
    points = [{ id: 5, x: 90, y: 620 }];
    await send("touchStart");
    points[0].y = 550;
    await send("touchMove");
    await expect(page.locator("[data-touch-stick]")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath(`${example}-phone-stick.png`) });
    points = [];
    await send("touchEnd");
    await expect(page.locator("[data-touch-stick]")).toBeHidden();
    expect(errors).toEqual([]);
    await context.close();
});
