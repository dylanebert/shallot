import { expect, test } from "playwright/test";

test.use({ baseURL: "http://127.0.0.1:4216" });

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
        const sample = (steps = 0) =>
            page.evaluate(
                (steps) =>
                    (
                        window as unknown as {
                            sample(steps: number): {
                                grounded: boolean;
                                position: number[];
                                yaw: number;
                                jumps: number;
                                use: boolean;
                            };
                        }
                    ).sample(steps),
                steps,
            );
        const session = await context.newCDPSession(page);
        let points: { id: number; x: number; y: number }[] = [];
        const send = async (type: "touchStart" | "touchMove" | "touchEnd") => {
            await session.send("Input.dispatchTouchEvent", { type, touchPoints: points });
            expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
        };
        await expect(page.getByRole("button", { name: "Jump", exact: true })).toBeHidden();
        if (claim === "move and look") {
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
            await expect(page.locator("[data-touch-stick]")).toBeVisible();
            points = [];
            await send("touchEnd");
            await expect(page.locator("[data-touch-stick]")).toBeHidden();
        } else if (claim === "jump") {
            // The first touch reveals the controls before the jump finger lands.
            points = [{ id: 1, x: 280, y: 450 }];
            await send("touchStart");
            const jump = await page
                .getByRole("button", { name: "Jump", exact: true })
                .boundingBox();
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
        const settled = await sample(120);
        const beforeKey = await sample(12);
        expect(
            Math.hypot(
                beforeKey.position[0] - settled.position[0],
                beforeKey.position[2] - settled.position[2],
            ),
            "no key must leave the settled player stationary",
        ).toBeLessThan(0.001);
        await page.keyboard.down("w");
        try {
            const now = await sample(12);
            expect(
                Math.hypot(
                    now.position[0] - beforeKey.position[0],
                    now.position[2] - beforeKey.position[2],
                ),
                "W must cause new horizontal movement",
            ).toBeGreaterThan(0.1);
        } finally {
            await page.keyboard.up("w");
        }
        expect(errors).toEqual([]);
        await context.close();
    });
}
