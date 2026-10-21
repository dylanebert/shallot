import { expect, test } from "playwright/test";

test.use({ baseURL: "http://127.0.0.1:4176" });

test("desktop instructions name keyboard and mouse", async ({ page }) => {
    await page.goto("/tests/touch.html");
    await expect
        .poll(() =>
            page.evaluate(
                () =>
                    "hint" in window &&
                    (window as unknown as { hint(value: string): boolean }).hint(
                        "WASD  Move\nSPACE  Jump\nMOUSE  Look",
                    ),
            ),
        )
        .toBe(true);
});

for (const claim of ["move and look", "jump"]) {
    test(`phone fingers ${claim} through the example's controls`, async ({ browser }) => {
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
                        sample(): {
                            grounded: boolean;
                            position: number[];
                            yaw: number;
                            jumps: number;
                            use: boolean;
                        };
                    }
                ).sample(),
            );
        const session = await context.newCDPSession(page);
        let points: { id: number; x: number; y: number }[] = [];
        const send = (type: "touchStart" | "touchMove" | "touchEnd") =>
            session.send("Input.dispatchTouchEvent", { type, touchPoints: points });
        await expect(page.getByRole("button", { name: "Jump", exact: true })).toBeHidden();
        if (claim === "move and look") {
            await expect
                .poll(() =>
                    page.evaluate(() =>
                        (window as unknown as { hint(value: string): boolean }).hint(
                            "WASD  Move\nSPACE  Jump\nMOUSE  Look",
                        ),
                    ),
                )
                .toBe(true);
            const before = await sample();
            points = [
                { id: 1, x: 90, y: 620 },
                { id: 2, x: 280, y: 450 },
            ];
            await send("touchStart");
            points[0].y = 550;
            points[1].x = 330;
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
            await expect.poll(async () => (await sample()).yaw).not.toBe(before.yaw);
            await expect
                .poll(() =>
                    page.evaluate(() =>
                        (window as unknown as { hint(value: string): boolean }).hint(
                            "STICK  Move\nDRAG  Look\nJump",
                        ),
                    ),
                )
                .toBe(true);
            await expect(page.locator("[data-touch-stick]")).toBeVisible();
            points = [];
            await send("touchEnd");
            await expect(page.locator("[data-touch-stick]")).toBeHidden();
        } else if (claim === "jump") {
            // The first touch reveals the controls before the jump finger lands.
            points = [{ id: 1, x: 280, y: 450 }];
            await send("touchStart");
            const button = page.getByRole("button", { name: "Jump", exact: true });
            await expect(button).toHaveCSS("background-color", "rgb(60, 50, 44)");
            await expect(button).toHaveCSS("border-color", "rgb(42, 36, 32)");
            await expect(button).toHaveCSS("font-family", '"IBM Plex Sans", system-ui, sans-serif');
            await expect(button).toHaveCSS("font-size", "13px");
            const jump = await button.boundingBox();
            await expect.poll(async () => (await sample()).grounded).toBe(true);
            const grounded = await sample();
            points.push({ id: 2, x: jump!.x + 32, y: jump!.y + 32 });
            await send("touchStart");
            await expect.poll(async () => (await sample()).jumps).toBeGreaterThan(grounded.jumps);
            await expect
                .poll(async () => (await sample()).position[1])
                .toBeGreaterThan(grounded.position[1] + 0.2);
            points = [];
            await send("touchEnd");
        }
        expect(errors).toEqual([]);
        await context.close();
    });
}
