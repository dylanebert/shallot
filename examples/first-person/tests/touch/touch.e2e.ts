import { type BrowserContext, expect, type Page, test } from "playwright/test";

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

test.describe(() => {
    let context: BrowserContext;
    let page: Page;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        context = await browser.newContext({
            viewport: { width: 390, height: 844 },
            hasTouch: true,
        });
        page = await context.newPage();
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto("/tests/touch.html?shared");
        await page.waitForFunction(() => "sample" in window);
    });
    test.afterAll(() => context.close());
    test.beforeEach(() =>
        page.evaluate(() => (window as unknown as { reset(): Promise<void> }).reset()),
    );
    for (const claim of ["move and look", "jump"]) {
        test(`phone fingers ${claim} through the example's controls`, async () => {
            const sample = (steps = 0, ticks = 0) =>
                page.evaluate(
                    ([steps, ticks]) =>
                        (
                            window as unknown as {
                                sample(
                                    steps: number,
                                    ticks: number,
                                ): {
                                    grounded: boolean;
                                    position: number[];
                                    yaw: number;
                                    jumps: number;
                                    use: boolean;
                                };
                            }
                        ).sample(steps, ticks),
                    [steps, ticks],
                );
            const session = await context.newCDPSession(page);
            let points: { id: number; x: number; y: number }[] = [];
            const send = async (type: "touchStart" | "touchMove" | "touchEnd") => {
                await session.send("Input.dispatchTouchEvent", { type, touchPoints: points });
                expect(await page.evaluate(() => document.activeElement === document.body)).toBe(
                    true,
                );
                await sample(1);
            };
            await expect(page.getByRole("button", { name: "Jump", exact: true })).toBeHidden();
            if (claim === "move and look") {
                await expect
                    .poll(
                        () =>
                            page.evaluate(() =>
                                (window as unknown as { hint(value: string): boolean }).hint(
                                    "WASD  Move\nSPACE  Jump\nMOUSE  Look",
                                ),
                            ),
                        { intervals: [0] },
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
                    .poll(
                        async () => {
                            const now = await sample(1);
                            return Math.hypot(
                                now.position[0] - before.position[0],
                                now.position[2] - before.position[2],
                            );
                        },
                        { intervals: [0] },
                    )
                    .toBeGreaterThan(0.1);
                await expect
                    .poll(async () => (await sample(1)).yaw, { intervals: [0] })
                    .not.toBe(before.yaw);
                await expect
                    .poll(
                        () =>
                            page.evaluate(() =>
                                (window as unknown as { hint(value: string): boolean }).hint(
                                    "STICK  Move\nDRAG  Look\nJump",
                                ),
                            ),
                        { intervals: [0] },
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
                await expect(button).toHaveCSS(
                    "font-family",
                    '"IBM Plex Sans", system-ui, sans-serif',
                );
                await expect(button).toHaveCSS("font-size", "13px");
                const jump = await button.boundingBox();
                await expect
                    .poll(async () => (await sample(1)).grounded, { intervals: [0] })
                    .toBe(true);
                const grounded = await sample();
                points.push({ id: 2, x: jump!.x + 32, y: jump!.y + 32 });
                await send("touchStart");
                await expect
                    .poll(async () => (await sample(1)).jumps, { intervals: [0] })
                    .toBeGreaterThan(grounded.jumps);
                await expect
                    .poll(async () => (await sample(1)).position[1], { intervals: [0] })
                    .toBeGreaterThan(grounded.position[1] + 0.2);
                points = [];
                await send("touchEnd");
            }
            // Settling and the keyless stretch need no input sampled, so they tick without drawing.
            const settled = await sample(0, 120);
            const beforeKey = await sample(0, 12);
            expect(
                Math.hypot(
                    beforeKey.position[0] - settled.position[0],
                    beforeKey.position[2] - settled.position[2],
                ),
                "no key must leave the settled player stationary",
            ).toBeLessThan(0.001);
            await page.keyboard.down("w");
            try {
                // Sample the held W once; the remaining fixed ticks read that same held input.
                const now = await sample(1, 11);
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
        });
    }
});
