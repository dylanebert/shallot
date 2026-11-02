import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "playwright/test";
import { pointerControl } from "../pointer";

test.use(pipelineWarnings);
for (const width of [1280, 390])
    test(`Dynamic Mover draws and accepts Walk, Aim and Shoot at ${width}`, async ({
        browser,
    }, info) => {
        const context = await browser.newContext({
            viewport: { width, height: width === 390 ? 844 : 720 },
            hasTouch: width === 390,
        });
        const page = await context.newPage();
        try {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            await page.goto(`${info.project.use.baseURL}/tests/controls.html?capture`);
            await page.waitForFunction(() => "dynamicState" in window);
            await pointerControl(page, page.getByRole("button", { name: "Scene", exact: true }));
            await pointerControl(
                page,
                page.getByRole("button", { name: "Dynamic Mover", exact: true }),
            );
            const state = (steps = 0) =>
                page.evaluate(
                    (steps) =>
                        (
                            window as unknown as {
                                dynamicState(steps: number): {
                                    keys: Record<string, number>;
                                    balls: number;
                                };
                            }
                        ).dynamicState(steps),
                    steps,
                );
            await state(2);
            if (width === 1280) {
                await page.mouse.click(640, 400);
                await expect
                    .poll(() => page.evaluate(() => !!document.pointerLockElement))
                    .toBe(true);
                await page.keyboard.down("Control");
                await state(1);
                expect((await state()).keys.Control).toBe(1);
                await page.keyboard.up("Control");
                await page.mouse.down({ button: "right" });
                await state(12);
                await expect(page.locator(".reticle")).toBeVisible();
                await page.mouse.down({ button: "left" });
                await state(1);
                expect((await state()).keys).toMatchObject({ Control: 0, Aim: 1, Shoot: 1 });
                expect((await state()).balls).toBe(1);
                await page.mouse.up({ button: "left" });
                await page.mouse.up({ button: "right" });
                await page.evaluate(() => document.exitPointerLock());
            } else {
                await page.touchscreen.tap(80, 600);
                await state(1);
                const session = await context.newCDPSession(page);
                const points: { id: number; x: number; y: number }[] = [];
                for (const [index, label] of ["Walk", "Aim", "Shoot"].entries()) {
                    const button = page.getByRole("button", { name: label, exact: true });
                    await expect(button).toBeVisible();
                    const box = (await button.boundingBox())!;
                    expect(box.width).toBeGreaterThanOrEqual(44);
                    expect(box.x + box.width).toBeLessThanOrEqual(390);
                    expect(box.y).toBeGreaterThan(0);
                    points.push({
                        id: index + 1,
                        x: box.x + box.width / 2,
                        y: box.y + box.height / 2,
                    });
                    await session.send("Input.dispatchTouchEvent", {
                        type: "touchStart",
                        touchPoints: points,
                    });
                    await state(1);
                }
                await expect(page.locator(".reticle")).toBeVisible();
                expect((await state()).keys).toMatchObject({ Control: 1, Aim: 1, Shoot: 1 });
                expect((await state()).balls).toBe(1);
                await session.send("Input.dispatchTouchEvent", {
                    type: "touchEnd",
                    touchPoints: [],
                });
                await session.detach();
            }
            await state(1);
            expect((await state()).keys).toMatchObject({ Control: 0, Aim: 0, Shoot: 0 });
            await expect(page.locator(".reticle")).toBeHidden();
            const frame = await page.evaluate(() =>
                (
                    window as unknown as {
                        drawDynamicFrame(): Promise<{
                            width: number;
                            height: number;
                            cyan: number;
                        }>;
                    }
                ).drawDynamicFrame(),
            );
            expect(frame.width).toBe(width);
            expect(frame.height).toBe(width === 390 ? 844 : 720);
            expect(frame.cyan).toBeGreaterThan(20);
            await page.evaluate(() => document.exitPointerLock());
            await expect.poll(() => page.evaluate(() => !!document.pointerLockElement)).toBe(false);
            await page.getByText("Physics", { exact: true }).click();
            const image = info.outputPath(`dynamic-mover-${width}.png`);
            await page.screenshot({ path: image });
            await info.attach(`dynamic-mover-${width}`, { path: image, contentType: "image/png" });
            expect(errors).toEqual([]);
        } finally {
            await context.close();
        }
    });
